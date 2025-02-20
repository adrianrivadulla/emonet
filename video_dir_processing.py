import os.path

from typing import List, Dict
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from skimage import io
from face_alignment.detection.sfd.sfd_detector import SFDDetector
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from emonet.models import EmoNet
import cv2
import pandas as pd


def load_video(video_path: Path) -> List[np.ndarray]:
    """
    Loads a video using OpenCV.
    """
    video_capture = cv2.VideoCapture(video_path)



    list_frames_rgb = []

    # Reads all the frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        list_frames_rgb.append(image_rgb)

    # Get fps
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Get video length in seconds
    video_len_s = len(list_frames_rgb) / fps

    return video_len_s, list_frames_rgb


def load_emonet(n_expression: int, device: str):
    """
    Loads the emotion recognition model.
    """

    # Loading the model
    state_dict_path = Path(__file__).parent.joinpath(
        "pretrained", f"emonet_{n_expression}.pth"
    )

    print(f"Loading the emonet model from {state_dict_path}.")
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    return net


def run_emonet(
    emonet: torch.nn.Module, frame_rgb: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Runs the emotion recognition model on a single frame.
    """
    # Resize image to (256,256)
    print(f"Running emonet on a frame of size {frame_rgb.shape}")
    image_rgb = cv2.resize(frame_rgb, (image_size, image_size))

    # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
    image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0

    with torch.no_grad():
        output = emonet(image_tensor.unsqueeze(0))

    return output


def plot_valence_arousal(
    valence: float, arousal: float, circumplex_size=512
) -> np.ndarray:
    """
    Assumes valence and arousal in range [-1;1].
    """
    circumplex_path = Path(__file__).parent / "images/circumplex.png"

    circumplex_image = cv2.imread(circumplex_path)
    circumplex_image = cv2.resize(circumplex_image, (circumplex_size, circumplex_size))

    # Position in range [0,circumplex_size/2] - arousal axis goes up, so need to take the opposite
    position = (
        (valence + 1.0) / 2.0 * circumplex_size,
        (1.0 - arousal) / 2.0 * circumplex_size,
    )

    cv2.circle(
        circumplex_image, (int(position[0]), int(position[1])), 16, (0, 0, 255), -1
    )

    return circumplex_image


def make_visualization(
    frame_rgb: np.ndarray,
    face_crop_rgb: np.ndarray,
    face_bbox: torch.Tensor,
    emotion_prediction: Dict[str, torch.Tensor],
    font_scale=2,
) -> np.ndarray:
    """
    Composes the final visualization with detected face, landmarks, discrete and continuous emotions.
    """
    # Visualize the detected face
    cv2.rectangle(
        frame_rgb,
        (face_bbox[0], face_bbox[1]),
        (face_bbox[2], face_bbox[3]),
        (255, 0, 0),
        8,
    )

    # Add the discrete emotion next to it
    predicted_emotion_class_idx = (
        torch.argmax(nn.functional.softmax(emotion_prediction["expression"], dim=1))
        .cpu()
        .item()
    )
    frame_rgb = cv2.putText(
        frame_rgb,
        emotion_classes[predicted_emotion_class_idx],
        ((face_bbox[0] + face_bbox[2]) // 2, face_bbox[1] + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Landmarks visualization
    # Resize to the original face_crop image size
    heatmap = torch.nn.functional.interpolate(
        emotion_prediction["heatmap"],
        (face_crop_rgb.shape[0], face_crop_rgb.shape[1]),
        mode="bilinear",
    )

    landmark_visualization = face_crop_rgb.copy()
    for landmark_idx in range(heatmap[0].shape[0]):
        # Detect the position of each landmark and draw a circle there
        landmark_position = (
            heatmap[0, landmark_idx, :, :] == torch.max(heatmap[0, landmark_idx, :, :])
        ).nonzero()
        cv2.circle(
            landmark_visualization,
            (
                int(landmark_position[0][1].cpu().item()),
                int(landmark_position[0][0].cpu().item()),
            ),
            4,
            (255, 255, 255),
            -1,
        )

    # Valence and arousal visualization
    circumplex_bgr = plot_valence_arousal(
        emotion_prediction["valence"].clamp(-1.0, 1.0),
        emotion_prediction["arousal"].clamp(-1.0, 1.0),
        frame_rgb.shape[0],
    )

    # Compose the final visualization
    visualization = np.zeros(
        (frame_rgb.shape[0], frame_rgb.shape[1] + frame_rgb.shape[0] // 2, 3),
        dtype=np.uint8,
    )

    # Resize the circumplex and face crop to match the frame size
    circumplex_bgr = cv2.resize(
        circumplex_bgr, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    landmark_visualization = cv2.resize(
        landmark_visualization, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    visualization[:, : frame_rgb.shape[1], :] = frame_rgb[:, :, ::-1].astype(np.uint8)
    visualization[
        : frame_rgb.shape[0] // 2, frame_rgb.shape[1] :, :
    ] = landmark_visualization[:, :, ::-1].astype(
        np.uint8
    )  # OpenCV needs BGR
    visualization[frame_rgb.shape[0] // 2 :, frame_rgb.shape[1] :, :] = (
        circumplex_bgr.astype(np.uint8)
    )

    return visualization


if __name__ == "__main__":

    # matplotlib.use("Qt5Agg")

    torch.backends.cudnn.benchmark = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nclasses",
        type=int,
        default=8,
        choices=[5, 8],
        help="Number of emotional classes to test the model on. Please use 5 or 8.",
    )
    # input video directory, default is a videos folder the current directory
    cwd = Path(__file__).parent
    parser.add_argument(
        "--video_dir_path",
        type=str,
        default=os.path.join(cwd, "example_videos"),
        help="Path to directory containing .mp4 videos to process.",
    )

    parser.add_argument(
        "--output_dir_path",
        type=str,
        default=os.path.join(cwd, "example_videos"),
        help="Path to directory where output videos and analysis .csvs will be saved.",
    )

    args = parser.parse_args()

    # Check if the video directory exists
    assert os.path.exists(args.video_dir_path), f"Input video directory {args.video_dir_path} does not exist."
    assert os.path.exists(args.output_dir_path), f"Output directory {args.output_dir_path} does not exist."

    # Parameters of the experiments
    n_expression = args.nclasses
    device = "cuda:0"
    # device = 'cpu'
    image_size = 256
    emotion_classes = {
        0: "Neutral",
        1: "Happy",
        2: "Sad",
        3: "Surprise",
        4: "Fear",
        5: "Disgust",
        6: "Anger",
        7: "Contempt",
    }

    print(f"Loading emonet")
    emonet = load_emonet(n_expression, device)

    print(f"Loading face detector")
    sfd_detector = SFDDetector(device)

    # Get video files in
    video_files = [f for f in os.listdir(args.video_dir_path) if f.endswith('.mp4') and 'emonet_output' not in f]
    assert len(video_files) > 0, f"No .mp4 files found in {args.video_dir_path}"

    print(f"Found {len(video_files)} .mp4 videos to process in {args.video_dir_path}")

    for vidi, video_file in enumerate(video_files):

        print(f"Loading ({vidi + 1}/{len(video_files)}) {video_file}")
        video_path = os.path.join(args.video_dir_path, video_file)
        video_len_s, list_frames_rgb = load_video(video_path)

        # Create a nan array of shape (n_frames, n_classes)
        emotion_prediction_output = np.ones((len(list_frames_rgb), n_expression + 2)) * np.nan

        visualization_frames = []

        for i, frame in enumerate(list_frames_rgb):

            # Run face detector
            with torch.no_grad():
                # Face detector requires BGR frame
                detected_faces = sfd_detector.detect_from_image(frame[:, :, ::-1])

            # If at least a face has been detected, run emotion recognition on the first face
            if len(detected_faces)>0:
                # Only take the first detected face
                bbox = np.array(detected_faces[0]).astype(np.int32)

                face_crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

                emotion_prediction = run_emonet(emonet, face_crop.copy())

                visualization_bgr = make_visualization(
                    frame.copy(), face_crop.copy(), bbox, emotion_prediction
                )
                visualization_frames.append(visualization_bgr)

                # Save the emotion prediction
                emotion_prediction_output[i, :-2] = nn.functional.softmax(
                    emotion_prediction["expression"], dim=1
                ).cpu().numpy()

                # Save arousal and valence values
                emotion_prediction_output[i, -2] = emotion_prediction["arousal"].cpu().numpy()
                emotion_prediction_output[i, -1] = emotion_prediction["valence"].cpu().numpy()

            else:
                # Visualization without emotion
                visualization = np.zeros(
                    (frame.shape[0], frame.shape[1] + frame.shape[0] // 2, 3),
                    dtype=np.uint8,
                )
                visualization[:, : frame.shape[1], :] = frame[:, :, ::-1].astype(np.uint8)

                visualization_frames.append(visualization)

            if i % 100 == 0:
                print(f"Ran prediction on {i}/{len(list_frames_rgb)} frames")

        # Write the result as a video
        if visualization_frames:
            save_path = os.path.join(args.output_dir_path, f"{video_file[:-4]}_emonet_output.mp4")

            out = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                24.0,
                (visualization_frames[0].shape[1], visualization_frames[0].shape[0]),
            )

            for frame in visualization_frames:
                out.write(frame)

            out.release()

        # Write the emotion predictions
        emotion_names = [emotion_classes[i] for i in range(n_expression)]
        df = pd.DataFrame(emotion_prediction_output, columns=[emotion_names + ["arousal", "valence"]])

        # Add frame column at the start
        df.insert(0, 't', range(len(df)))

        # Make t column time in seconds
        df['t'] = df['t'] * (video_len_s / len(df))

        df.to_csv(save_path[:-4] + ".csv", index=False, float_format='%.4f')
        print(f"Finished ({vidi + 1}/{len(video_files)}). Video saved at {save_path}")
        print(f"Emotion predictions saved at {save_path[:-4]}.csv")

    print("All videos processed.")
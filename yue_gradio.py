import gradio as gr
import subprocess
import os
import json
import tempfile
import sys
from pathlib import Path

def check_cuda_setup():
    """Check CUDA setup and return appropriate device"""
    import torch
    if not torch.cuda.is_available():
        return None, "CUDA is not available. GPU is required for this model."
    
    try:
        # Test CUDA memory allocation
        device = torch.device('cuda')
        test_tensor = torch.zeros((1, 1)).to(device)
        del test_tensor
        return device, None
    except Exception as e:
        return None, f"CUDA error: {str(e)}"

def check_requirements():
    """Check if all requirements are met"""
    messages = []
    
    # Check if git-lfs is installed
    try:
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        messages.append("Git LFS is not installed. Please install it using:\n"
                       "apt-get install git-lfs && git lfs install")
    
    # Check if model files are downloaded
    semantic_ckpt_path = Path("inference/xcodec_mini_infer/semantic_ckpts/hf_1_325000")
    if not semantic_ckpt_path.exists():
        messages.append("Model checkpoints not found. Please run:\n"
                       "git lfs install && git lfs pull")
    
    # Check CUDA setup
    _, cuda_error = check_cuda_setup()
    if cuda_error:
        messages.append(cuda_error)
    
    return messages

def load_top_200_tags():
    try:
        with open('top_200_tags.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return ["pop", "rock", "electronic"]  # Default fallback tags

def generate_music(
    mode,
    genre_tags,
    lyrics,
    run_n_segments,
    stage2_batch_size,
    max_new_tokens,
    audio_prompt=None,
    prompt_start_time=0,
    prompt_end_time=30,
    progress=gr.Progress()
):
    # Check requirements first
    requirement_messages = check_requirements()
    if requirement_messages:
        raise gr.Error("\n".join(requirement_messages))
    
    if not genre_tags.strip():
        raise gr.Error("Genre tags cannot be empty")
    
    if not lyrics.strip():
        raise gr.Error("Lyrics cannot be empty")
    
    progress(0, desc="Creating temporary files...")
    
    # Create temporary files for genre and lyrics
    genre_path = None
    lyrics_path = None
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as genre_file:
            genre_file.write(genre_tags)
            genre_path = genre_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as lyrics_file:
            lyrics_file.write(lyrics)
            lyrics_path = lyrics_file.name

        # Ensure output directory exists
        output_dir = Path("inference/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clean previous output files
        for file in output_dir.glob("*.wav"):
            try:
                file.unlink()
            except:
                pass

        # Base command with environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        cmd = [
            sys.executable,  # Use the current Python interpreter
            "infer.py",
            "--stage2_model", "m-a-p/YuE-s2-1B-general",
            "--genre_txt", genre_path,
            "--lyrics_txt", lyrics_path,
            "--run_n_segments", str(run_n_segments),
            "--stage2_batch_size", str(stage2_batch_size),
            "--output_dir", "./output",
            "--cuda_idx", "0",
            "--max_new_tokens", str(max_new_tokens)
        ]

        # Add model based on mode
        if mode == "Chain of Thought (CoT)":
            cmd.extend(["--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-cot"])
        else:  # ICL mode
            if not audio_prompt:
                raise gr.Error("Reference audio is required for ICL mode")
            cmd.extend([
                "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-icl",
                "--audio_prompt_path", audio_prompt.name,
                "--prompt_start_time", str(prompt_start_time),
                "--prompt_end_time", str(prompt_end_time)
            ])

        progress(0.2, desc="Starting music generation...")
        
        # Run the command
        try:
            process = subprocess.run(
                cmd, 
                check=True, 
                cwd="inference",
                capture_output=True,
                text=True,
                env=env
            )
            progress(0.8, desc="Processing output...")
            
            # Find the generated audio file in output directory
            output_files = list(output_dir.glob("*.wav"))
            if output_files:
                return str(output_files[-1]), None
            else:
                raise gr.Error("No audio file was generated")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Error during music generation:\n{e.stderr}"
            if "CUDA out of memory" in error_msg:
                error_msg += "\n\nTry reducing the batch size or number of segments."
            raise gr.Error(error_msg)
            
    except Exception as e:
        if isinstance(e, gr.Error):
            raise e
        raise gr.Error(f"An error occurred: {str(e)}")
    finally:
        # Cleanup temporary files
        for path in [genre_path, lyrics_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

def create_interface():
    # Load available tags
    available_tags = load_top_200_tags()
    
    with gr.Blocks(title="YuE Music Generation") as interface:
        gr.Markdown("# YuE Music Generation Interface")
        gr.Markdown("Generate music from lyrics using YuE model")
        
        # Add system status
        status_messages = check_requirements()
        if status_messages:
            with gr.Box(variant="stop"):
                gr.Markdown("### ⚠️ System Requirements Not Met")
                for msg in status_messages:
                    gr.Markdown(f"- {msg}")
        
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    choices=["Chain of Thought (CoT)", "In-Context Learning (ICL)"],
                    label="Generation Mode",
                    value="Chain of Thought (CoT)"
                )
                
                genre_tags = gr.TextArea(
                    label="Genre Tags",
                    placeholder="Enter genre tags separated by space (e.g., inspiring female uplifting pop airy vocal)",
                    info="Select from available tags below"
                )
                
                gr.Markdown("### Available Tags:")
                gr.Dataframe(
                    headers=["Tags"],
                    value=[[tag] for tag in available_tags],
                    label="Available Tags"
                )
                
                lyrics = gr.TextArea(
                    label="Lyrics",
                    placeholder="Enter lyrics with structure labels (e.g., [verse], [chorus])",
                    info="Separate sections with double newlines"
                )
                
                with gr.Row():
                    run_n_segments = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="Number of Segments"
                    )
                    stage2_batch_size = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Stage 2 Batch Size"
                    )
                    max_new_tokens = gr.Slider(
                        minimum=1000,
                        maximum=6000,
                        value=3000,
                        step=100,
                        label="Max New Tokens"
                    )

                with gr.Group() as icl_options:
                    audio_prompt = gr.Audio(
                        label="Reference Audio (for ICL mode)",
                        type="filepath"
                    )
                    with gr.Row():
                        prompt_start = gr.Number(
                            value=0,
                            label="Prompt Start Time (seconds)"
                        )
                        prompt_end = gr.Number(
                            value=30,
                            label="Prompt End Time (seconds)"
                        )

            with gr.Column():
                generate_btn = gr.Button("Generate Music", variant="primary")
                output_audio = gr.Audio(label="Generated Music")
                error_output = gr.Markdown(visible=False)

        # Show/hide ICL options based on mode
        mode.change(
            fn=lambda x: gr.Group.update(visible=(x == "In-Context Learning (ICL)")),
            inputs=[mode],
            outputs=[icl_options]
        )

        # Generate button click event
        generate_btn.click(
            fn=generate_music,
            inputs=[
                mode,
                genre_tags,
                lyrics,
                run_n_segments,
                stage2_batch_size,
                max_new_tokens,
                audio_prompt,
                prompt_start,
                prompt_end
            ],
            outputs=[output_audio, error_output]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0"
    ) 

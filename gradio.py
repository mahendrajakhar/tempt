import gradio as gr
import subprocess
import os
import json
import tempfile

def load_top_200_tags():
    with open('top_200_tags.json', 'r') as f:
        return json.load(f)

def generate_music(
    mode,
    genre_tags,
    lyrics,
    run_n_segments,
    stage2_batch_size,
    max_new_tokens,
    audio_prompt=None,
    prompt_start_time=0,
    prompt_end_time=30
):
    # Create temporary files for genre and lyrics
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as genre_file:
        genre_file.write(genre_tags)
        genre_path = genre_file.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as lyrics_file:
        lyrics_file.write(lyrics)
        lyrics_path = lyrics_file.name

    # Base command
    cmd = [
        "python", "infer.py",
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
        cmd.extend([
            "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-icl",
            "--audio_prompt_path", audio_prompt.name if audio_prompt else "",
            "--prompt_start_time", str(prompt_start_time),
            "--prompt_end_time", str(prompt_end_time)
        ])

    try:
        # Run the command
        subprocess.run(cmd, check=True, cwd="inference")
        
        # Find the generated audio file in output directory
        output_files = os.listdir("inference/output")
        audio_files = [f for f in output_files if f.endswith('.wav')]
        if audio_files:
            return os.path.join("inference/output", audio_files[-1])
        else:
            return "No audio file generated"
    finally:
        # Cleanup temporary files
        os.unlink(genre_path)
        os.unlink(lyrics_path)

def create_interface():
    # Load available tags
    available_tags = load_top_200_tags()
    
    with gr.Blocks(title="YuE Music Generation") as interface:
        gr.Markdown("# YuE Music Generation Interface")
        gr.Markdown("Generate music from lyrics using YuE model")
        
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
            outputs=output_audio
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True
        ) 

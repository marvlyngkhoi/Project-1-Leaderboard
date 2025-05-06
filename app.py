import gradio as gr
import json
import os
import pandas as pd
import time

# Configuration
FOLDER_TO_WATCH = "model_results"  # Folder where JSON files are stored
REFRESH_INTERVAL = 5  # Seconds between automatic refreshes
COLUMNS_TO_DISPLAY = ["model_name", "accuracy", "precision", "recall", "f1_score", "timestamp"]

# Create the folder if it doesn't exist
os.makedirs(FOLDER_TO_WATCH, exist_ok=True)

def load_data():
    data = []
    for filename in os.listdir(FOLDER_TO_WATCH):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(FOLDER_TO_WATCH, filename), 'r') as f:
                    model_data = json.load(f)
                    data.append(model_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if data:
        df = pd.DataFrame(data)
        if not df.empty and 'accuracy' in df.columns:
            df = df.sort_values(by='accuracy', ascending=False)
        return df[COLUMNS_TO_DISPLAY] if all(col in df.columns for col in COLUMNS_TO_DISPLAY) else df
    return pd.DataFrame(columns=COLUMNS_TO_DISPLAY)

def update_leaderboard():
    current_data = load_data()
    return current_data, time.strftime("%Y-%m-%d %H:%M:%S")

with gr.Blocks(title="Model Leaderboard") as demo:
    gr.Markdown("# Model Performance Leaderboard")
    gr.Markdown(f"Automatically updates from JSON files in `{FOLDER_TO_WATCH}` folder")
    
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        last_update = gr.Textbox(label="Last Updated", interactive=False)
    
    leaderboard = gr.Dataframe(
        headers=COLUMNS_TO_DISPLAY,
        interactive=False,
        wrap=True
    )
    
    refresh_btn.click(
        fn=update_leaderboard,
        outputs=[leaderboard, last_update]
    )
    
    # New way to implement auto-refresh
    demo.load(
        fn=lambda: update_leaderboard(),
        outputs=[leaderboard, last_update]
    )
    
    # Alternative auto-refresh method
    def auto_refresh():
        while True:
            time.sleep(REFRESH_INTERVAL)
            yield update_leaderboard()
    
    demo.queue()
    leaderboard.change(
        fn=auto_refresh,
        outputs=[leaderboard, last_update],
        show_progress=False
    )

if __name__ == "__main__":
    demo.launch(share=True)
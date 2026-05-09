# Program title: Kids Image Storytelling App

# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image


# -------------------------------
# Function part
# -------------------------------

# Image to text
def img2text(image):
    image_to_text_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

    text = image_to_text_model(image)[0]["generated_text"]
    return text


# Text to story
def text2story(text):
    story_pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    prompt = (
        "Write a warm, positive, simple English story for kids aged 3 to 10. "
        "The story should be 50 to 100 words. "
        "Use friendly language and a happy ending. "
        f"The story is based on this image description: {text}"
    )

    story_results = story_pipe(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    story_text = story_results[0]["generated_text"]

    # Remove the prompt if it appears in the output
    story_text = story_text.replace(prompt, "").strip()

    # Keep the story under 100 words
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100]) + "."

    return story_text


# Text to audio
def text2audio(story_text):
    audio_pipe = pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )

    audio_data = audio_pipe(story_text)
    return audio_data



# -------------------------------
# Main part
# -------------------------------

st.set_page_config(
    page_title="Kids Image Storytelling App",
    page_icon="🧸"
)

st.title("🧸 Kids Image Storytelling App")

st.write(
    "Upload an image, and this app will create a short story and read it aloud!"
)

uploaded_file = st.file_uploader(
    "Please upload an image:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Step 1: Image to text
    st.subheader("Image Caption")
    with st.spinner("Understanding the image..."):
        scenario = img2text(image)

    st.write(f"**Scenario:** {scenario}")

    # Step 2: Text to story
    st.subheader("Generated a Story")
    with st.spinner("Creating a story for kids..."):
        story = text2story(scenario)

    st.write(f"**Story:** {story}")

    # Step 3: Text to audio
    st.subheader("Story Audio")
    with st.spinner("Generating audio..."):
        audio_data = text2audio(story)

    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]

    st.audio(audio_array, sample_rate=sample_rate)

else:
    st.info("Please upload an image to start.")

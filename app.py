# Program title: Kids Image Storytelling App

# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image


# -------------------------------
# Load models
# -------------------------------

@st.cache_resource
def load_image_captioning_model():
    """
    Load the image captioning model.
    This model turns an image into a short text description.
    """
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_generation_model():
    """
    Load the story generation model.
    This model turns the image description into a short story.
    """
    return pipeline(
        "text-generation",
        model="pranavpsv/genre-story-generator-v2"
    )


@st.cache_resource
def load_text_to_audio_model():
    """
    Load the text-to-audio model.
    This model turns the story into speech.
    """
    return pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )


# -------------------------------
# Function part
# -------------------------------

def img2text(image):
    """
    This function takes an uploaded image and generates a caption.
    """
    image_to_text_model = load_image_captioning_model()
    result = image_to_text_model(image)
    text = result[0]["generated_text"]
    return text


def text2story(text):
    """
    This function takes the image caption and generates a short story.
    The story is designed for children aged 3 to 10.
    """
    story_model = load_story_generation_model()

    prompt = (
        "Write a warm, positive, simple English story for kids aged 3 to 10. "
        "The story should be 50 to 100 words. "
        "Use friendly language and a happy ending. "
        f"The story is based on this image description: {text}"
    )

    story_results = story_model(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    story_text = story_results[0]["generated_text"]

    # Remove the prompt from the generated result if it appears
    story_text = story_text.replace(prompt, "").strip()

    # Make sure the story is not longer than 100 words
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100]) + "."

    return story_text


def text2audio(story_text):
    """
    This function converts the generated story into audio.
    """
    audio_model = load_text_to_audio_model()
    audio_data = audio_model(story_text)
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
    st.subheader("Step 1: Image Caption")
    with st.spinner("Understanding the image..."):
        caption = img2text(image)

    st.write("**Image Caption:**")
    st.write(caption)

    # Step 2: Text to story
    st.subheader("Step 2: Generated Story")
    with st.spinner("Creating a story for kids..."):
        story = text2story(caption)

    st.write("**Story:**")
    st.write(story)

    # Step 3: Text to audio
    st.subheader("Step 3: Story Audio")
    with st.spinner("Generating audio..."):
        audio_data = text2audio(story)

    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]

    st.audio(audio_array, sample_rate=sample_rate)

else:
    st.info("Please upload an image to start.")

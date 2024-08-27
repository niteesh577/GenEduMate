import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure the Gemini API
# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')



def check_code_with_gemini(user_code, exercise_solution):
    """Check user code against the correct solution using Gemini API."""
    prompt = f"User's code:\n{user_code}\n\nExpected solution:\n{exercise_solution}\n\nEvaluate the user's code for correctness, identify the programming language, and suggest improvements if any."
    response = model.generate_content(prompt)
    return response

def display_exercise_page():
    """Display interactive exercises with feedback and progress tracking."""
    st.header("Interactive Coding Exercises")

    # Sample coding exercises data
    exercises = {
        "Exercise 1": {
            "description": "Write a function that returns the factorial of a number.",
            "solution": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
        },
        "Exercise 2": {
            "description": "Write a function that checks if a number is prime.",
            "solution": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        }
    }

    # Exercise selection
    exercise_name = st.selectbox("Select an Exercise", options=list(exercises.keys()))
    exercise = exercises[exercise_name]

    # Display exercise description
    st.subheader("Description")
    st.write(exercise["description"])

    # Code input area
    st.subheader("Your Solution")
    user_code = st.text_area("Write your code here", height=200)

    # Submit button
    if st.button("Submit"):
        if user_code:
            response = check_code_with_gemini(user_code, exercise["solution"])
            st.write(response.text)
            if "correct" in response.text.lower():
                st.success("Great job! Your solution is correct.")
            else:
                st.error("Your solution is incorrect. Please try again.")
        else:
            st.warning("Please enter your code before submitting.")

def main():
    st.set_page_config(page_title="Interactive Coding Exercises", layout="wide")
    display_exercise_page()

if __name__ == "__main__":
    main()
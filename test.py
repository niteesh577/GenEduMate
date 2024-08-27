if page == "Home":
    st.header("Welcome to the Personalized Learning Platform")
    st.write("Here you can ask questions about the PDF files and take practice quizzes.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # Extract the response text
        response_text = response.get("output_text", "Sorry, I couldn't find an answer.")

        # Display the answer
        st.write("Reply: ", response_text)

        if response_text != "Sorry, I couldn't find an answer.":
            st.subheader("Practice Quiz")

            # Generate initial quiz questions
            text_chunks = get_text_chunks(response_text)


            def generate_quiz(text_chunks):
                quizzes = []
                for chunk in text_chunks:
                    sentences = chunk.split('. ')
                    if len(sentences) > 1:
                        question = sentences[0] + "?"
                        correct_answer = sentences[1]
                        # Generate some random incorrect answers
                        incorrect_answers = generate_incorrect_answers(correct_answer, text_chunks)
                        options = [correct_answer] + incorrect_answers
                        random.shuffle(options)  # Shuffle the options

                        quiz = {
                            'question': question,
                            'options': options,
                            'correct_answer': correct_answer
                        }
                        quizzes.append(quiz)
                return quizzes


            def generate_incorrect_answers(correct_answer, text_chunks):
                incorrect_answers = []
                for chunk in text_chunks:
                    sentences = chunk.split('. ')
                    for sentence in sentences:
                        if sentence and sentence != correct_answer and len(incorrect_answers) < 3:
                            incorrect_answers.append(sentence)
                            if len(incorrect_answers) == 3:
                                break
                return incorrect_answers


            quizzes = generate_quiz(text_chunks)


            def run_quiz(quizzes):
                quiz_completed = False
                user_answers = []

                for idx, quiz in enumerate(quizzes, 1):
                    st.write(f"Quiz {idx}: {quiz['question']}")
                    for option_idx, option in enumerate(quiz['options'], 1):
                        st.write(f"{option_idx}. {option}")

                    user_answer = st.text_input(f"Your answer for Quiz {idx}", key=f"answer_quiz_{idx}")
                    if user_answer:
                        user_answers.append(user_answer)
                        correct_answer = quiz['correct_answer']
                        is_correct = evaluate_answer(user_answer, correct_answer)
                        if is_correct:
                            st.write("Correct! 🎉")
                        else:
                            st.write("Incorrect. Please try again.")

                        # Provide follow-up questions based on user answers
                        follow_up_prompt = f"Generate follow-up questions based on the user's response: {user_answer}."
                        try:
                            follow_up_response = genai.generate_text(prompt=follow_up_prompt)
                            if follow_up_response and hasattr(follow_up_response, 'text'):
                                follow_up_questions = [q.strip() for q in follow_up_response.text.split('\n') if
                                                       q.strip()]

                                if follow_up_questions:
                                    for follow_up_idx, follow_up_question in enumerate(follow_up_questions, 1):
                                        st.write(f"Follow-up Question {follow_up_idx}: {follow_up_question}")
                                        follow_up_answer = st.text_input(
                                            f"Your answer for Follow-up Question {follow_up_idx}",
                                            key=f"followup_answer_{idx}_{follow_up_idx}")
                                        if follow_up_answer:
                                            follow_up_clarification = genai.generate_text(
                                                prompt=f"Clarify the follow-up answer: {follow_up_answer}")
                                            if follow_up_clarification and hasattr(follow_up_clarification, 'text'):
                                                st.write("Clarification: ", follow_up_clarification.text)
                                            else:
                                                st.error("The follow-up clarification does not contain text.")
                            else:
                                st.error("The follow-up response does not contain text.")
                        except Exception as e:
                            st.error(f"An error occurred while generating follow-up questions: {str(e)}")

                if len(user_answers) == len(quizzes):
                    quiz_completed = True

                return quiz_completed, user_answers


            quiz_completed = False
            while not quiz_completed:
                quiz_completed, user_answer = run_quiz(quizzes)
                if not quiz_completed and user_answer:
                    follow_up_prompt = f"Generate follow-up questions based on the user's response: {user_answer}."
                    try:
                        follow_up_response = genai.generate_text(prompt=follow_up_prompt)
                        if follow_up_response and hasattr(follow_up_response, 'text'):
                            follow_up_questions = [q.strip() for q in follow_up_response.text.split('\n') if
                                                   q.strip()]
                            quizzes = follow_up_questions  # Update quizzes with follow-up questions
                            st.write("Here are some follow-up questions to help you understand better.")
                        else:
                            st.error("The follow-up response does not contain text.")
                    except Exception as e:
                        st.error(f"An error occurred while generating follow-up questions: {str(e)}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.pdf_docs = pdf_docs
                st.success("Documents processed successfully!")
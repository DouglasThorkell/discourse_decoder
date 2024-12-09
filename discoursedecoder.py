import streamlit as st
import os
import time
import requests
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from sentence_transformers import SentenceTransformer, util
from langchain.schema import HumanMessage, SystemMessage
import re
import subprocess
import sqlite3

# Ensure the updated SQLite is installed
if tuple(map(int, sqlite3.sqlite_version.split("."))) < (3, 35, 0):
    st.write("Updating SQLite...")
    result = subprocess.run(["bash", "setup.sh"], capture_output=True, text=True)
    st.write(result.stdout)
    st.write(result.stderr)
else:
    st.write(f"SQLite version is already up-to-date: {sqlite3.sqlite_version}")

## Get Valid URL Function ##
def get_valid_url():
    url = st.text_input("Enter the URL of a webpage: ").strip()
    if not url:
        return None
    if not re.match(r'^https?://', url):
        st.error("Invalid URL. Please ensure it starts with http or https.")
        return None
    return url

## Fetch URL Function ##
def fetch_url_content(url, retries=3, backoff_factor=1):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            st.warning(f"**Attempt {attempt + 1}/{retries} failed:** {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                st.error("**Error fetching the URL:** All attempts failed. Please check the URL or try again later.")
                return None

## Paragraph Based Chunking Function ##
def paragraph_based_chunking(raw_text, max_chunk_size=3000, overlap=500):
    paragraphs = raw_text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph[-overlap:] + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [Document(page_content=chunk) for chunk in chunks]

## Initializing Retrieval Chain Function ##
def initialize_retrieval_chain_from_text(raw_text):
    chunks = paragraph_based_chunking(raw_text)

    openai_api_key = os.environ.get('openai_api_key')
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    llm = get_llm(temperature=0.7, model="gpt-4o-mini")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    crchain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return crchain

## Extracting Main Arguments Function ##
def extract_main_arguments(retrieval_chain):
    prompt = (
        "Summarize the main argument in the document in 100 words or less."
        "Your summary should:"
        "Start with the title of the article in bold and within quotation marks, followed by a blank line."
        "Highlight the core claims made by the author."
        "Include the key evidence or examples that support these claims."
        "Use formal, objective language. Begin with an introductory sentence that captures the essence of the article, "
        "followed by the supporting points and conclusion. For example:"
        "'The article argues that [main argument], supported by [key evidence]. It concludes by [conclusion].'"
    )

    result = retrieval_chain({"question": prompt})

    return result["answer"]

## Extracting Three Stances Function ##
def extract_three_stances(retrieval_chain):
    prompts = {
        "supportive": "Create a user-centered statement that supports the argument in the article."
                      "It should be a simple stance that is easily differentiated from other viewpoints."
                      "Your answer should be concise and within quotation marks. You are the user."
                      "For example, say: 'I am pro-choice, which means I believe that a woman has the right to make decisions about her own body without interference, including the choice to have an abortion.'"
                      "Start with: 'I am', 'I believe', or 'I think'."
                      "Your answer should be around 30 words.",
        "opposing": "Create a user-centered statement that opposes the argument in the article."
                    "It should be a simple stance that is easily differentiated from other viewpoints."
                    "Your answer should be concise and within quotation marks. You are the user."
                    "For example, say: 'I am pro-choice, which means I believe that a woman has the right to make decisions about her own body without interference, including the choice to have an abortion.'"
                     "Start with: 'I am', 'I believe', or 'I think'."
                     "Your answer should be around 30 words.",
        "middle": "Create a user-centered statement that niether supports nor opposes the argument in the article."
                  "It should be a simple stance that cannot decide between a supportive or an opposing view of the article."
                  "Your answer should be concise and within quotation marks. You are the user."
                  "For example, say: 'I am pro-choice, which means I believe that a woman has the right to make decisions about her own body without interference, including the choice to have an abortion.'"
                  "Start with: 'I am', 'I believe', or 'I think'."
                  "Your answer should be around 30 words.",
    }

    fallback_responses = {
        "supportive": '"I believe there are strong points in the article’s perspective that deserve further exploration, and I generally lean toward agreeing with its stance."',
        "opposing": '"I believe there are valid concerns about the arguments presented, and I am inclined to challenge its conclusions."',
        "middle": '"I believe it’s important to consider all sides of the discussion, and I find myself seeing value in multiple perspectives without fully committing to one."'
    }

    results = {}
    for key, prompt in prompts.items():
        response = retrieval_chain({"question": prompt})["answer"]
        st.write(f"Raw Response for `{key}`: {response}")

        if ":" in response and response.lower().startswith("a user-centered statement"):
            response = response.split(":", 1)[-1].strip()
        cleaned_response = response.strip().strip('"')

        if "I don't know" in response or cleaned_response.startswith("The") or cleaned_response.startswith("While"):
            results[key] = fallback_responses[key]
        else:
            results[key] = f'"{cleaned_response}"'

    return {"pro": results["supportive"], "con": results["opposing"], "middle": results["middle"]}

## OpenAI API ##
openai_api_key = os.environ.get('OPENAI_API_KEY')
os.environ['openai_api_key'] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

## Serper API ##
serper_api_key = os.environ.get('SERPER_API_KEY')
os.environ["SERPER_API_KEY"] = serper_api_key

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

## Supporting Argument Crew ##
supporting_writer = Agent(
    role="Supporting Writer",
    goal="Develop a concise argument supporting {topic}.",
    backstory="You are tasked with crafting a clear, persuasive argument in favor of {topic}.",
    allow_delegation=False,
    verbose=False
)
supporting_researcher = Agent(
    role="Supporting Researcher",
    goal="Provide relevant evidence supporting {topic} to strengthen the Writer's argument.",
    backstory="You expand the Writer's argument by finding relevant examples, statistics, or case studies.",
    allow_delegation=False,
    verbose=False,
    tools=[search_tool, scrape_tool]
)
supporting_debater = Agent(
    role="Supporting Debater",
    goal="Combine the Writer's argument and the Researcher's evidence into a clear, final 100-word argument supporting {topic}.",
    backstory="You are responsible for presenting the Supporting Crew's final argument in a concise and compelling manner.",
    allow_delegation=False,
    verbose=False
)
supporting_writer_task = Task(
    description="Draft a clear, persuasive argument in favor of {topic} that is concise and provides a foundation for adding supporting evidence. Address any questions posed by the Moderator Crew: {moderator_question}.",
    expected_output="A concise argument supporting {topic} that the Researcher can expand on with evidence.",
    agent=supporting_writer,
    async_execution=True
)
supporting_researcher_task = Task(
    description= "Expand on the Writer's argument with evidence such as statistics, examples, or case studies "
                 "that support {topic}. Use the search tool or scrape tool to find supporting evidence on the topic {topic}. "
                 "Choose the most relevant source for high-quality evidence."
                 "Ensure the evidence addresses any questions posed by the Moderator Crew: {moderator_question}.",
    expected_output="Relevant evidence in text form that strengthens the Writer's argument.",
    agent=supporting_researcher,
    async_execution=True
)
supporting_debater_task = Task(
    description="Combine the Writer's argument and the Researcher's evidence to create a 100-word final argument supporting {topic}. Ensure the argument addresses any questions posed by the Moderator Crew: {moderator_question}.",
    expected_output="A finalized, polished 100-word argument supporting {topic}.",
    agent=supporting_debater
)
supporting_crew = Crew(
    agents=[supporting_writer, supporting_researcher, supporting_debater],
    tasks=[supporting_writer_task, supporting_researcher_task, supporting_debater_task],
    verbose=False
)

## Opposing Argument Crew ##
opposing_writer = Agent(
    role="Opposing Writer",
    goal="Develop a concise counterargument against {topic}, responding directly to the Supporting Crew's argument: {supporting_argument}.",
    backstory="You are tasked with crafting a strong counterargument based on the Supporting Crew's argument.",
    allow_delegation=False,
    verbose=False
)
opposing_researcher = Agent(
    role="Opposing Researcher",
    goal="Provide relevant evidence opposing {topic}, directly addressing the Supporting Crew's argument: {supporting_argument}.",
    backstory="You expand the Writer's counterargument by finding evidence, examples, or statistics that directly challenge the Supporting Crew's points.",
    allow_delegation=False,
    verbose=False,
    tools=[search_tool, scrape_tool]
)
opposing_debater = Agent(
    role="Opposing Debater",
    goal="Combine the Writer's counterargument and the Researcher's evidence into a clear, final 100-word argument opposing {topic}.",
    backstory="You are responsible for presenting the Opposing Crew's final argument in a concise and compelling manner.",
    allow_delegation=False,
    verbose=False
)
opposing_writer_task = Task(
    description="Draft a clear counterargument opposing {topic} that directly addresses the Supporting Crew's argument: {supporting_argument}. Also, address any questions posed by the Moderator Crew: {moderator_question}.",
    expected_output="A concise counterargument opposing {topic} that the Researcher can expand on with evidence.",
    agent=opposing_writer,
    async_execution=True
)
opposing_researcher_task = Task(
    description="Expand on the Writer's counterargument with evidence such as statistics, examples, or case studies that challenge the Supporting Crew's argument: {supporting_argument}."
                "Use the search tool or scrape tool to find supporting evidence on the topic {topic}. "
                "Choose the most relevant source for high-quality evidence."
                "Ensure the evidence also addresses any questions posed by the Moderator Crew: {moderator_question}.",
    expected_output="Relevant evidence in text form that strengthens the Writer's counterargument.",
    agent=opposing_researcher,
    async_execution=True
)
opposing_debater_task = Task(
    description="Combine the Writer's counterargument and the Researcher's evidence to create a 100-word final argument opposing {topic}. Ensure the argument addresses any questions posed by the Moderator Crew: {moderator_question}.",
    expected_output="A finalized, polished 100-word counterargument opposing {topic}.",
    agent=opposing_debater
)
opposing_crew = Crew(
    agents=[opposing_writer, opposing_researcher, opposing_debater],
    tasks=[opposing_writer_task, opposing_researcher_task, opposing_debater_task],
    verbose=False
)

## Moderator Crew ##
moderator = Agent(
    role="Moderator",
    goal="Provide a 100-word summary evaluating arguments from both Supporting and Opposing Crews.",
    backstory="You impartially summarize the round, considering the Supporting Crew's argument: {supporting_argument} and the Opposing Crew's counterargument: {opposing_argument}.",
    allow_delegation=False,
    verbose=False
)
question_analyst = Agent(
    role="Question Analyst",
    goal="Pose a neutral question addressing both crews' arguments: {supporting_argument} and {opposing_argument}, unless it is the last round.",
    backstory="You generate a question to deepen the discussion based on the arguments presented by both crews unless it is the last round.",
    allow_delegation=False,
    verbose=False
)
summary_writer = Agent(
    role="Summary Writer",
    goal="Craft a 100-word reflection summarizing the key points from the round and explicitly include the question generated by the Question Analyst unless it is the last round.",
    backstory="You produce a concise summary based on the Supporting Crew's argument: {supporting_argument}, the Opposing Crew's counterargument: {opposing_argument}, and the question generated by the Question Analyst unless it is the last round.",
    allow_delegation=False,
    verbose=False
)
moderator_task = Task(
    description="Summarize and reflect on arguments from both crews: {supporting_argument} and {opposing_argument} in 100 words. Include the question generated by the Question Analyst at the end of the summary unless it's the last round.",
    expected_output="100-word neutral summary including the question generated by the Question Analyst.",
    agent=moderator
)
question_analyst_task = Task(
    description="Generate a neutral question based on arguments presented: {supporting_argument} and {opposing_argument}, unless it is the last round.",
    expected_output="A neutral, thought-provoking question based on the arguments.",
    agent=question_analyst
)
summary_writer_task = Task(
    description=(
        "Produce a concise reflection summarizing the key points from the Supporting Crew ({supporting_argument}) "
        "and Opposing Crew ({opposing_argument}). Include the question generated by the Question Analyst at the end "
        "of your output explicitly labeled as '**Moderator's Question:**'."
    ),
    expected_output=(
        "100-word summary of the debate round, with the question appended as '**Moderator's Question:**', unless it is the last round."
    ),
    agent=summary_writer
)
moderation_crew = Crew(
    agents=[moderator, question_analyst, summary_writer],
    tasks=[moderator_task, question_analyst_task, summary_writer_task],
    verbose=False
)

## Finding Common Ground Function ##
def find_common_ground_debate(supporting_arguments, opposing_arguments, similarity_threshold=0.7):
    if not supporting_arguments or not opposing_arguments:
        return "No significant common ground could be identified across the debate rounds."

    ## Sentence Transformer Model ##
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        return f"Error loading model: {str(e)}"

    all_supporting_sentences = []
    all_opposing_sentences = []
    for round_num, (supporting, opposing) in enumerate(zip(supporting_arguments, opposing_arguments), 1):
        all_supporting_sentences += [
            f"Round {round_num}: {sentence}" for sentence in supporting.split(". ") if sentence
        ]
        all_opposing_sentences += [
            f"Round {round_num}: {sentence}" for sentence in opposing.split(". ") if sentence
        ]

    try:
        supporting_embeddings = model.encode(all_supporting_sentences, convert_to_tensor=True)
        opposing_embeddings = model.encode(all_opposing_sentences, convert_to_tensor=True)
    except Exception as e:
        return f"Error computing embeddings: {str(e)}"

    similarity_matrix = util.cos_sim(supporting_embeddings, opposing_embeddings)

    common_ground_pairs = []
    for i, supporting_sentence in enumerate(all_supporting_sentences):
        for j, opposing_sentence in enumerate(all_opposing_sentences):
            if similarity_matrix[i][j] >= similarity_threshold:
                pair = (
                    supporting_sentence.split(": ")[-1],
                    opposing_sentence.split(": ")[-1],
                    similarity_matrix[i][j].item()
                )
                common_ground_pairs.append(pair)

    if common_ground_pairs:
        themes = [pair[0] for pair in common_ground_pairs]
        unique_themes = list(set(themes))

        llm = get_llm(temperature=0.7, model="gpt-4o-mini")

        prompt = (
            "The following are themes of agreement identified across multiple debate rounds:\n\n"
            + "\n".join(f"- {theme}" for theme in unique_themes)
            + "\n\n"
            "Based on these identified themes, craft a cohesive and concise statement summarizing the common ground between opposing views. Your summary should:"
            "Clearly articulate the shared principles or areas of agreement."
            "Acknowledge the distinctions or key differences in the opposing perspectives."
            "Explain how the common ground perspective incorporates elements from both sides to foster a unified understanding."
            "Use formal, balanced language and avoid overly technical jargon. Ensure the statement is clear, actionable, and limited to 100 words."
            "For example: 'Despite differing views on [opposing points], both sides agree on [common principles]. This shared understanding emphasizes [implication of agreement].'"
        )

        try:
            messages = [
                SystemMessage(content="You are an AI assistant that helps find common ground."),
                HumanMessage(content=prompt),
            ]
            response = llm(messages)

            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return "Unexpected response format from the LLM."
        except Exception as e:
            return f"Error generating response from LLM: {str(e)}"
    else:
        return "No significant common ground could be identified across the debate rounds."

## Final Insight Function ##
def generate_final_insight(common_ground):
    if "common_ground" not in st.session_state:
        st.session_state.common_ground = None

    if "No significant common ground" in common_ground:
        return (
            "Based on the debate, no significant overlaps were identified. This indicates "
            "strongly polarized views, and further discussion may require finding shared "
            "underlying values or fostering a deeper dialogue."
        )

    if not common_ground or not isinstance(common_ground, str) or not common_ground.strip():
        return "Error: Invalid common ground input."

    try:
        llm = get_llm(temperature=0.7, model="gpt-4o-mini")
    except Exception as e:
        return f"Error initializing the LLM: {str(e)}"

    prompt = (
        f"The following statement summarizes the common ground across multiple debate rounds:\n\n"
        f"{common_ground}\n\n"
        "Using the identified common ground, provide practical suggestions for an individual moving forward."
        "Your recommendations should:"
        "Be tailored to the themes in the common ground statement."
        "Focus on impactful words and actions that the individual can take to foster understanding and reconciliation."
        "Use a direct tone, addressing the individual as 'you.'"
        "End with a specific recommendation for how the individual should engage when discussing the topic with someone who holds an opposing viewpoint."
        "Begin this final sentence with: 'When talking to someone about [topic], you should...'"
        "Limit your response to 50 words and ensure it is concise, actionable, and user-centered."
    )

    try:
        from langchain.schema import HumanMessage

        messages = [
            HumanMessage(content=prompt)
        ]
        response = llm(messages)
    except Exception as e:
        return f"Error generating response from LLM: {str(e)}"

    if hasattr(response, "content"):
        final_insight = response.content
    elif isinstance(response, str):
        final_insight = response
    else:
        final_insight = "Unexpected response format from the LLM."

    return final_insight

## Main Function ##
def main():

    st.title("Discourse Decoder")

    with st.sidebar:
        st.title("About")
        st.markdown(

            """
            **Discourse Decoder** is a tool designed to analyze and break down arguments from online content.

            **It Helps You**:

            - Extract main arguments from a webpage.
            - Dynamically engage in debates with supportive, opposing, and neutral perspectives.
            - Identify common ground and actionable insights.

            **How It Works:**

            1. Paste the URL of a webpage into the input field above.
            2. Wait for the application to fetch and analyze the content.
            3. Review the article arguments and select a stance to simulate a debate.
            4. See how the debate unfolds.
            5. Review the common ground and find actionable insights.
            """
        )

    ## Step 1: Getting a valid URL from the user ##
    url = get_valid_url()
    if not url:
        return

    ## Step 2: Fetching content from the URL ##
    with st.spinner("Fetching content..."):
        raw_content = fetch_url_content(url)
    if not raw_content:
        st.error("Failed to fetch content from the URL. Exiting...")
        return
    st.success("Content fetched successfully!")

    ## Step 3: Initializing retrieval chain ##
    retrieval_chain = initialize_retrieval_chain_from_text(raw_content)

    ## Step 4: Extracting main arguments dynamically ##
    if "article_arguments" not in st.session_state:
        with st.spinner("Extracting arguments..."):
            st.session_state.article_arguments = extract_main_arguments(retrieval_chain)
    st.subheader("Article")
    st.markdown(f"{st.session_state.article_arguments}")

    ## Step 5: Extracting stances dynamically ##
    if "stances" not in st.session_state:
        with st.spinner("Extracting stances..."):
            st.session_state.stances = extract_three_stances(retrieval_chain)

    stances = st.session_state.stances

    st.subheader("Select a stance")
    st.markdown("Select a stance based on the analysis:")
    st.markdown(f"1. **Supportive:** {stances['pro']}")
    st.markdown(f"2. **Opposing:** {stances['con']}")
    st.markdown(f"3. **Neutral:** {stances['middle']}")

    user_choice = st.radio("Enter the number of your choice:", ["1", "2", "3"])

    if st.button("Confirm Choice"):
        st.session_state.user_stance = (
            stances["pro"] if user_choice == "1" else stances["con"] if user_choice == "2" else stances["middle"]
        )
        st.success(f"You selected: {st.session_state.user_stance}")

        ## Step 6: Debate Simulation ##
        st.subheader("Debate Simulation")

        moderator_question = None
        debate_transcript = []
        supporting_arguments = []
        opposing_arguments = []

        for round_num in range(1, 4):
            with st.expander(f"Round {round_num}", expanded=True):
                st.markdown(f"### **Round {round_num}**")

                with st.spinner(f"Processing Round {round_num}..."):

                    ## Supporting Argument Crew ##
                    supporting_inputs = {
                        "topic": st.session_state.article_arguments,
                        "objective": "Highlight the benefits",
                        "moderator_question": moderator_question or ""
                    }
                    if round_num > 1:
                        supporting_inputs["opposing_arguments"] = opposing_result.raw
                    supporting_result = supporting_crew.kickoff(supporting_inputs)
                    supporting_arguments.append(supporting_result.raw.strip())

                    ## Opposing Argument Crew ##
                    opposing_inputs = {
                        "topic": st.session_state.user_stance,
                        "objective": "Highlight the risks",
                        "supporting_argument": supporting_result.raw,
                        "moderator_question": moderator_question or ""
                    }
                    opposing_result = opposing_crew.kickoff(opposing_inputs)
                    opposing_arguments.append(opposing_result.raw.strip())

                    ## Moderator Crew ##
                    moderation_inputs = {
                        "topic": st.session_state.article_arguments,
                        "supporting_argument": supporting_result.raw,
                        "opposing_argument": opposing_result.raw
                    }
                    moderation_result = moderation_crew.kickoff(moderation_inputs)

                    if round_num <= 3 and "**Moderator's Question:**" in moderation_result.raw:
                        moderator_question = moderation_result.raw.split("**Moderator's Question:**")[-1].strip()
                    else:
                        moderator_question = None

                    moderator_output_clean = moderation_result.raw.replace(
                        f"**Moderator's Question:** {moderator_question}", "").strip()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### **Article View**")
                        st.markdown(f"> {supporting_result.raw.strip()}")

                    with col2:
                        st.markdown("#### **Your View**")
                        st.markdown(f"> {opposing_result.raw.strip()}")

                    st.markdown("#### **Moderator's Perspective**")
                    st.markdown(f"{moderator_output_clean}")

                    debate_transcript.append(f"**Round {round_num}**")
                    debate_transcript.append(f"**Supporting View:** {supporting_result.raw.strip()}")
                    debate_transcript.append(f"**Opposing View:** {opposing_result.raw.strip()}")
                    debate_transcript.append(f"**Moderator View:** {moderator_output_clean}")

                    if round_num < 3:
                        debate_transcript.append(f"**Moderator's Question for Next Round:** {moderator_question}")
                        st.markdown(f"#### **Moderator's Question for Next Round**")
                        st.markdown(f"{moderator_question}")

        ## Step 7: Find Common Ground ##
        st.subheader("Potential Avenues for Reconciliation")
        with st.spinner("Analyzing common ground..."):
            common_ground = find_common_ground_debate(supporting_arguments, opposing_arguments)
        st.markdown(f"**Common Ground Identified:** {common_ground}")

        ## Step 8: Generating Final Insight ##
        with st.spinner("Creating final insight..."):
            final_insight = generate_final_insight(common_ground)
        st.subheader("Final Insight")
        st.markdown(f"{final_insight}")

if __name__ == "__main__":
    main()

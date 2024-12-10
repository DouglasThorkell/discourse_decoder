# Discourse Decoder
### Purpose, Methodologies and Aspirations

Discourse Decoder is a tool designed to analyze and break down arguments from online content. As demonstrated by a recent project from Google Deepmind,**[1]** gathering a demographically representative sample of the population in the United Kingdom, AI-generated statements used for the purpose of mediation were, in fact, preferred by a majority of study participants. Through the use of an AI mediator, small and polarized groups of participants became less divided and more inclined to listen to opposing views. It is upon this idea I decided to build what would become the Discourse Decoder. 

## Purpose

The Discourse Decoder is designed to foster constructive dialogue and empowers users to meaningfully engage with diverse perspectives, facilitating understanding and reconciliation in discussions surrounding contentious topics. It addresses the critical need for constructive public discourse in an environment where polarization has become the norm. The primary objectives of the Discourse Decoder are to:

	Encourage Critical Thinking: It enables users to challenge or validate their perspectives by interacting with AI agents simulating supportive, opposing, and neutral viewpoints.

	Promote Dialogue and Consensus: It goes beyond polarized debates by identifying common ground and actionable solutions between contrasting views.

	Demonstrate Knowledge About LLMs: It showcases the integration of advanced Large Language Models (LLM) technologies, to a large extent through LangChain,**[2]** including effective prompt engineering, retrieval-augmented generation (RAG), conversational retrieval chains, AI agents using AI crew, and sentence transformers to perform sentiment analysis. 
 
	Provide a User-Friendly Environment. It leverages a simple user interface (UI), hosted locally through the Streamlit library and application,**[3]** to not only enhance the critical thinking, dialogue and consensus and technical aspects of the application, but 

In practice, the Discourse Decoder helps the user to extract main arguments from a webpages, dynamically engage in debates with supportive, opposing, and neutral perspectives, and identify common ground and actionable insights. It works by, in order: 

1.	Prompting the user to paste URL of webpage into the input field at the top of the page when launching the application;
2.	Waiting for the application to fetch and analyze the content in the URL provided;
3.	Reviewing the URL’s arguments and allowing the user to pick one out of three user-centered stances;
4.	Initializing a debate between two LLMs mediated by another LLM moderator, each of the debate participants arguing in favour of the viewpoints furthered in the provided URL and the stance selected by the user, respectively.
5.	Continuing the simulated debate for three consecutive rounds, where the moderator is allowed to comment on and ask questions to the debate participants.
6.	Analyzing the debate rounds upon completion, looking to identify common ground between the two perspectives.
7.	Visualizing the relative three-dimensional space between arguments given during the debate as a network graph, showing similarity between the different arguments.
8.	Providing actionable insights to the user and giving a recommendation on how to effectively discuss the topic in the URL with those who do not hold the same view as the user.

## Notable Features

#### 1.	Common Ground Identification.
One of the platform's main features the ability to identify shared themes or values between opposing perspectives. The techniques used to implement this include:
 
	Semantic analysis employed as sentence embeddings and cosine similarity to detect thematic overlaps in arguments.
	Network visualizations meant to highlight points of agreement and divergence, providing a visual representation of the application’s more prominent debate dynamics.
	AI-generated suggestions implemented to simulate real-world discourse proposes reconciliation strategies, encouraging users to consider actionable compromises.

#### 2.	Technologies and Methodologies
The application leverages a robust AI ecosystem to deliver seamless functionality and a rich user experience. These include:  

	Core technologies, such as: 

o	Prompt engineering ensures precise argument extraction and debate structuring.

o	LangChain orchestrates multi-agent workflows and retrieval processes.

o	RAG (retrieval-augmented generation) enhances argument quality by integrating external evidence into AI-generated content. One example of this is the use of searching and scraping the internet using the Serper API key.

o	Semantic Analysis detects thematic overlaps using embedding models (where a personal API key was used to utilize OpenAI and a free publicly accessible API key as used for SentenceTransformers).

o	Visualization tools such as NetworkX and Plotly generate interactive visualizations for argument mapping

	Methodologies, such as: 

o	AI agent crews where a supporting crew (for instance, consisting of a writer, researcher, and debater) develops compelling arguments in favor of the uploaded article, an opposing crew crafts counterarguments that challenge the article's stance, and a moderation crew overseeing the debate flow to identifies common ground and provide neutral commentary.

o	AI-driven debate, where the debate unfolds over three structured rounds, with each round advancing the discussion through rebuttals and counterarguments. Arguments evolve as agents build on prior rounds, ensuring that the dialogue remains coherent and progressively deeper. Moreover, the stance customization which shapes the AI agents' interactions and debate dynamics.

## Value Proposition

The Discourse Decoder offers a unique combination of argumentation and reconciliation recommendations, making it a valuable tool for a wide range of applications. Three of these, provided that a degree of optimism surrounding the capabilities of the application, are:

	Education. The application fosters critical thinking skills by simulating AI-generated debates on a range of topics. Through features such as the generation of a final insight, students and those enrolled in continuing education can acquire the practical reconciliation skills needed in a polarized post-truth society.

	Media Literacy. The application helps users evaluate information critically, which enables informed decision-making when talking about and forming opinions on current events.
 
	Professional Negotiation: The application aids those who want to identify mutually beneficial solutions in high-stakes discussions. The ‘Avenues for Reconciliation’ and ‘Visualizing Argument Similarity’ sections are particularly useful in this endeavour. 

Overall, the platform’s potential in bridging divides and fostering understanding sets it apart from traditional debate tools, emphasizing not just competition but collaboration and growth.

## Bibliography

**[1]** Tessler, M., Bakker, M., Jarrett, A., Sheahan, H., Chadwick, M., Koster, R. & Summerfield, C. (2024). AI can help humans find common ground in democratic deliberation. Science, 386(6719), eadq2852.

**[2]** LangChain. (n.d.) LangChain Documentation. Available at: https://langchain.readthedocs.io/ (Accessed: 9 December 2024).

**[3]** Streamlit. (n.d.) Streamlit Documentation. Available at: https://docs.streamlit.io/ (Accessed: 9 December 2024).

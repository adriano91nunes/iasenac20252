# Importa as classes necessárias da CrewAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import SerperDevTool

# Importa as configurações do ambiente (chave de API)
import os
from dotenv import load_dotenv
load_dotenv()

# Configura a chave de API para o Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Inicializa a ferramenta de busca, que será usada pelo Analista de Dados
serper_search_tool = SerperDevTool()

# --- Definição dos Agentes ---

# 1. Agente Analista de Dados de Livros
analyst_agent = Agent(
    role="Analista de Dados de Livros",
    goal="Coletar e analisar dados de tendências de leitura, resenhas e popularidade de livros em diversas plataformas.",
    backstory="""Com uma vasta memória e acesso a bases de dados da internet, 
                este agente se especializa em encontrar padrões e descobrir o que as pessoas 
                estão realmente lendo e gostando. Ele busca a verdade por trás dos números, 
                garantindo que as indicações sejam não apenas populares, mas de alta qualidade.""",
    tools=[serper_search_tool],  # Atribui a ferramenta de busca a este agente
    verbose=True, # Mostra o que o agente está pensando e fazendo
    allow_delegation=False, # Não permite que este agente delegue tarefas
    llm=os.getenv("GROQ_API_KEY") # Usaremos o Groq como LLM
)

# 2. Agente Especialista em Gêneros Literários
genre_specialist_agent = Agent(
    role="Especialista em Gêneros Literários",
    goal="""Com base na análise de dados, identificar um gênero literário específico 
            e detalhar as características, autores de destaque e as nuances que o tornam atraente.""",
    backstory="""Com uma paixão inigualável pela literatura, este agente é um profundo 
                conhecedor de todos os gêneros. Ele consegue capturar a essência de cada 
                categoria, entendendo as expectativas dos leitores para traduzir a análise 
                de dados em um contexto literário.""",
    verbose=True,
    allow_delegation=False,
    llm=os.getenv("GROQ_API_KEY")
)

# 3. Agente Roteirista Criativo
scriptwriter_agent = Agent(
    role="Roteirista Criativo",
    goal="""Combinar a análise de dados e o conhecimento de gêneros para escrever um roteiro 
            de vídeo ou post de blog envolvente sobre as indicações de livros.""",
    backstory="""Este é o agente responsável por transformar dados e fatos em uma narrativa 
                convincente. Com um talento natural para a escrita, ele cria roteiros que parecem 
                ter sido escritos por um verdadeiro crítico literário, usando uma linguagem vibrante 
                para descrever o enredo e o impacto emocional das obras.""",
    verbose=True,
    allow_delegation=False,
    llm=os.getenv("GROQ_API_KEY")
)

# --- Definição das Tarefas ---

# A tarefa do Analista, que inicia o processo
task_analyst = Task(
    description="""Pesquisar as tendências atuais e os livros mais populares no nicho de ficção científica, 
                   analisando resenhas de usuários e notas em plataformas como Goodreads e Amazon. 
                   Liste pelo menos 5 títulos relevantes com uma breve justificativa para cada um.""",
    agent=analyst_agent,
    expected_output="Uma lista formatada dos 5 livros mais populares em ficção científica, com nome do livro, autor, e uma breve justificativa. O texto deve ser detalhado e conciso."
)

# A tarefa do Especialista, que usa a saída do Analista
task_genre_specialist = Task(
    description="""Com base na lista de livros fornecida, descreva as características do gênero 
                   de ficção científica, destacando por que esses livros são bons exemplos e quem se 
                   beneficiaria com a leitura. O resultado deve ser um parágrafo que contextualiza a lista.""",
    agent=genre_specialist_agent,
    expected_output="Um parágrafo conciso sobre o gênero de ficção científica, contextualizando a lista de livros gerada anteriormente."
)

# A tarefa do Roteirista, que finaliza o processo
task_scriptwriter = Task(
    description="""Utilizando a lista de livros e o contexto de gênero, redija um roteiro completo 
                   para um post de blog com o título 'Os 5 Livros de Ficção Científica que Você Precisa Ler'. 
                   O roteiro deve ser envolvente, persuasivo e incluir uma breve sinopse para cada livro, 
                   além de um 'gancho' para motivar a leitura.""",
    agent=scriptwriter_agent,
    expected_output="Um roteiro completo para post de blog, com título, introdução envolvente e descrição detalhada de cada um dos 5 livros."
)

# --- Criação e Execução da Crew ---

# Cria o objeto Crew, com os agentes e as tarefas
book_crew = Crew(
    agents=[analyst_agent, genre_specialist_agent, scriptwriter_agent],
    tasks=[task_analyst, task_genre_specialist, task_scriptwriter],
    process=Process.sequential,  # Define que as tarefas serão executadas em sequência
    verbose=True # Mostra o progresso e a saída final
)

# Inicia o processo da Crew
result = book_crew.kickoff()

# Imprime o resultado final
print("\n--- Roteiro de Indicação de Livros ---")
print(result)

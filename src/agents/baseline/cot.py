from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

class ChainOfThoughtAgent:
    """
    Ultra-simple chain of thought agent with direct LLM invoke
    """
    
    def __init__(self, model: str, temperature: float = 0.7):
        self.llm = ChatOllama(model=model, temperature=temperature)
    
    def generate_response(self, query: str, context: str = "") -> str:
        """
        Generate a chain of thought response to the given query
        
        Args:
            query: The problem or question to solve
            context: Optional context information to help with reasoning
            
        Returns:
            String containing the step-by-step reasoning and answer
        """

        system_prompt = """You are a helpful assistant that solves problems using step-by-step chain of thought reasoning.

Always break down your thinking into clear steps:
1. Understand what the problem is asking
2. Identify the key information given (including any context)
3. Work through the solution logically step by step
4. State your final answer clearly within "<answer>...</answer>"

Think through each step carefully and show your work."""

        user_message = f"""
{context}

Query: {query}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)

        print(response.content)
        
        # Parse the response to extract only the answer
        full_response = response.content
        
        # Find answer tags and extract content
        start_tag = "<answer>"
        end_tag = "</answer>"
        
        if start_tag in full_response and end_tag in full_response:
            start_idx = full_response.find(start_tag) + len(start_tag)
            end_idx = full_response.find(end_tag)
            return full_response[start_idx:end_idx].strip()
        else:
            # Fallback if tags not found
            return full_response


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = ChainOfThoughtAgent(model="mistral:7b")
    
    # Test with context
    context = """""sentences": [
[
"Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956.",
" Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\".",
" From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS.",
" Despite the program's long run, fewer than 24 episodes are known to exist."
],
[
"Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.",
" As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."
],
[
"Janet Marie Waldo (February 4, 1920 – June 12, 2016) was an American radio and voice actress.",
" She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."
],
[
"Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954.",
" The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."
],
[
"The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707.",
" A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."
],
[
"A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale.",
" It stars Shirley Temple in her final starring role as well as her final film appearance.",
" It is a sequel to the 1945 film \"Kiss and Tell\".",
" \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence.",
" The film was released on November 25, 1949, by United Artists."
],
[
"Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.",
" In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys.",
" The parents' bickering about which girl is the worse influence causes more problems than it solves."
],
[
"The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003.",
" Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State.",
" At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."
],
[
"The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent.",
" The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\".",
" The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra.",
" The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra.",
" The position was known as the \"shanbogaru\" in South Karnataka."
],
[
"Charles Craft (May 9, 1902 – September 19, 1968) was an English-born American film and television editor.",
" Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927.",
" The first film he edited was the Universal Pictures silent film, \"Painting the Town\".",
" Over the next 25 years, Craft would edit 90 feature-length films.",
" In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951–53, for which he was the main editor, editing 93 of the 98 episodes.",
" He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955–56), and \"Highway Patrol\" (1955–57).",
" In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes.",
" His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\".",
" When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966.",
" Craft died on September 19, 1968 in Los Angeles, California."
]
]"""
    
    query = 'What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?'
    
    print("Context:", context)
    print(f"Question: {query}\n")
    
    response = agent.generate_response(query, context)
    print("Agent Response:")
    print(response)
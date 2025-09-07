from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import re

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

REQUIRED RESPONSE FORMAT:
You must structure your response using these XML tags:

<reasoning>
Provide your step-by-step thinking process:
1. Understand what the problem is asking
2. Identify the key information given (including any context)
3. Work through the solution logically step by step
</reasoning>

<answer>
Provide your final answer here
</answer>

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
        
        # Parse the response to extract the answer
        response_content = response.content
        
        # Find the LAST occurrence of answer tags to get the final answer
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, response_content, re.DOTALL)
        
        if matches:
            # Get the last match (final answer)
            response_content = matches[-1].strip()
        else:
            # If no answer tags found, return the full response
            response_content = response_content.strip()
        
        return {"content": response_content}


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = ChainOfThoughtAgent(model="mistral:7b")
    
    # Test with context
    context = """""[""['The Andre Norton Award for Young Adult Science Fiction and Fantasy is an annual award presented by the Science Fiction and Fantasy Writers of America (SFWA) to the author of the best young adult or middle grade science fiction or fantasy book published in the United States in the preceding year.', ' It is named to honor prolific science fiction and fantasy author Andre Norton (1912–2005), and it was established by then SFWA president Catherine Asaro and the SFWA Young Adult Fiction committee and announced on February 20, 2005.', ' Any published young adult or middle grade science fiction or fantasy novel is eligible for the prize, including graphic novels.', ' There is no limit on word count.', ' The award is presented along with the Nebula Awards and follows the same rules for nominations and voting; as the awards are separate, works may be simultaneously nominated for both the Andre Norton award and a Nebula Award.']"", '[\'Victoria Hanley is an American young adult fantasy novelist.\', \' Her first three books, ""The Seer And The Sword"", ""The Healer\\\'s Keep"" and ""The Light Of The Oracle"" are companion books to one another.\', \' Her newest book (released March 2012) is the sequel of a series, called ""Indigo Magic"", published by Egmont USA.\', \' She\\\'s also published two non-fiction books through Cotton Wood Press; called ""Seize the Story: A Handbook For Teens Who Like To Write"", and ""Wild Ink: A Grownups Guide To Writing Fiction For Teens"".\']', '[\'The Hork-Bajir Chronicles is the second companion book to the ""Animorphs"" series, written by K. A. Applegate.\', \' With respect to continuity within the series, it takes place before book #23, ""The Pretender"", although the events told in the story occur between the time of ""The Ellimist Chronicles"" and ""The Andalite Chronicles"".\', \' The book is introduced by Tobias, who flies to the valley of the free Hork-Bajir, where Jara Hamee tells him the story of how the Yeerks enslaved the Hork-Bajir, and how Aldrea, an Andalite, and her companion, Dak Hamee, a Hork-Bajir, tried to save their world from the invasion.\', \' Jara Hamee\\\'s story is narrated from the points of view of Aldrea, Dak Hamee, and Esplin 9466, alternating in similar fashion to the ""Megamorphs"" books.\']', '[\'Shadowshaper is a 2015 American urban fantasy young adult novel written by Daniel José Older.\', \' It follows Sierra Santiago, an Afro-Boricua teenager living in Brooklyn.\', \' She is the granddaughter of a ""shadowshaper"", or a person who infuses art with ancestral spirits.\', \' As forces of gentrification invade their community and a mysterious being who appropriates their magic begins to hunt the aging shadowshapers, Sierra must learn about her artistic and spiritual heritage to foil the killer.\']', '[\'""Left Behind: The Kids (stylized as LEFT BEHIND >THE KIDS<)"" is a series written by Jerry B. Jenkins, Tim LaHaye, and Chris Fabry.\', \' The series consists of 40 short novels aimed primarily at the young adult market based on the adult series Left Behind also written by Jerry B. Jenkins.\', \' It follows a core group of teenagers as they experience the rapture and tribulation, based on scriptures found in the Bible, and background plots introduced in the adult novels.\', \' Like the adult series, the books were published by Tyndale House Publishing, and released over the 7 year period of 1997-2004.\', \' The series has sold over 11 million copies worldwide.\']', '[\'Dozens of Square Enix companion books have been produced since 1998, when video game developer Square began to produce books that focused on artwork, developer interviews, and background information on the fictional worlds and characters in its games rather than on gameplay details.\', \' The first series of these books was the ""Perfect Works"" series, written and published by Square subsidiary DigiCube.\', \' They produced three books between 1998 and 1999 before the line was stopped in favor of the ""Ultimania"" (アルティマニア , Arutimania ) series, a portmanteau of ultimate and mania.\', \' This series of books is written by Studio BentStuff, which had previously written game guides for Square for ""Final Fantasy VII"".\', \' They were published by DigiCube until the company was dissolved in 2003.\', \' Square merged with video game publisher Enix on April 1, 2003 to form Square Enix, which resumed publication of the companion books.\']', '[\'The Divide trilogy is a fantasy young adult novel trilogy by Elizabeth Kay, which takes place in an alternate universe.\', \' The three books are ""The Divide"" (2002), ""Back to The Divide"" (2005), and ""Jinx on The Divide"" (2006).\', \' The first novel was originally published by the small press publisher Chicken House (now a division of Scholastic), with subsequent volumes published by Scholastic, which also reprinted the first novel.\', \' The books have been translated into French, German, Spanish, Finnish, Chinese, Japanese, Portuguese, Italian, Romanian and Dutch.\', \' Interior illustrations are by Ted Dewan.\']', '[\'Science Fantasy, which also appeared under the titles Impulse and SF Impulse, was a British fantasy and science fiction magazine, launched in 1950 by Nova Publications as a companion to Nova\\\'s ""New Worlds"".\', \' Walter Gillings was editor for the first two issues, and was then replaced by John Carnell, the editor of ""New Worlds"", as a cost-saving measure.\', \' Carnell edited both magazines until Nova went out of business in early 1964.\', \' The titles were acquired by Roberts & Vinter, who hired Kyril Bonfiglioli to edit ""Science Fantasy""; Bonfiglioli changed the title to ""Impulse"" in early 1966, but the new title led to confusion with the distributors and sales fell, though the magazine remained profitable.\', \' The title was changed again to ""SF Impulse"" for the last few issues.\', \' ""Science Fantasy"" ceased publication the following year, when Roberts & Vinter came under financial pressure after their printer went bankrupt.\']', ""['Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic.', ' It is told in first person, with all six main characters taking turns narrating the books through their own perspectives.', ' Horror, war, dehumanization, sanity, morality, innocence, leadership, freedom and growing up are the core themes of the series.']"", ""['Etiquette & Espionage is a young adult steampunk novel by Gail Carriger.', ' It is her first young adult novel, and is set in the same universe as her bestselling Parasol Protectorate adult series.']""]"""
    
    query = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
    
    print("Context:", context)
    print(f"Question: {query}\n")
    
    response = agent.generate_response(query, context)
    print("Agent Response:")
    print(response)
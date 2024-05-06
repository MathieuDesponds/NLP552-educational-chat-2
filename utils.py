from typing import Any, Dict, Union
import gpt_wrapper
from gpt_wrapper.chat import Chat
from IPython import display
from tqdm import tqdm

gpt_wrapper.api_key = "a0e78a0b-9ef6-4d71-a054-75b743c03de7"#Mathieu : "a0e78a0b-9ef6-4d71-a054-75b743c03de7", Antoine :"fc8f2ffd-9b78-401e-a77d-162512bb597c"

import json

def load_data(path):
    with open(path, 'rb') as f :
        data = json.load(f)
    return data

def format_json(data_point, chat, message):
    data_point = data_point.copy()
    data_point['chat_id'] = chat.to_dict()['chat_id']
    data_point['interaction'] = [{
        'role' : message.to_dict()['role'],
        'content' : message.to_dict()['content']
    }]
    return data_point

def save_(data, at):
    with open(at, "w") as f:
        json.dump(data, f)

class GPTQuerier:
    
    def __init__(self, technique):
        self.technique = technique
        self.session_id = 1000

        self.prompt_QQ = lambda sample : f"""Here is the question : {sample['question']}. """
        answers = lambda sample : ' '.join([f"""{i+1}. "{answer}" """ for i, answer in enumerate(sample['choices'])])
        self.prompt_MCQ = lambda sample : f"""
        Here is the question : {sample['question']}. 
        Here are the answers : {answers(sample)}.
        """
        if(technique == 0):
            self.instruction_prompt_QQ = """
                You'll have a multiple choice question. Give only the final answer without context.
            """
            self.instruction_prompt_MCQ = self.instruction_prompt_QQ
        elif(technique == 1):
            self.instruction_prompt_QQ = """
                Answer this question by giving the answer in one sentence.
            """
            self.instruction_prompt_MCQ = self.instruction_prompt_QQ
        elif(technique == 2):
            self.instruction_prompt_QQ = """
                Answer this question by giving the answer in one sentence.
            """
            self.instruction_prompt_MCQ = self.instruction_prompt_QQ
        elif(technique == 3):
            self.instruction_prompt_QQ = """
                Answer this question by giving the answer in one sentence.
            """
            self.instruction_prompt_MCQ = self.instruction_prompt_QQ
        elif(technique == 4):
            self.instruction_prompt_QQ = """
                Answer this question by giving the answer in one sentence.
            """
            self.instruction_prompt_MCQ = self.instruction_prompt_QQ
            
        elif(technique == 'Mathieu'):
            self.prompt_QQ = PROMPT_MATHIEU_QQ
            self.prompt_MCQ = PROMPT_MATHIEU_MCQ
            self.instruction_prompt_QQ = PROMPT_INSTRUCTION_MATHIEU_QQ
            self.instruction_prompt_MCQ = PROMPT_INSTRUCTION_MATHIEU_MCQ
            
        elif(technique == 'Antoine'):
            self.instruction_prompt_QQ = PROMPT_INSTRUCTION_ANTOINE
            self.instruction_prompt_MCQ = self.instruction_prompt_QQ
        
        elif(technique == 'Mehron'):
            self.prompt_QQ = PROMPT_MEHRON_QQ
            self.prompt_MCQ = PROMPT_MEHRON_MCQ
            self.instruction_prompt_QQ = ""
            self.instruction_prompt_MCQ = ""
            
    @staticmethod
    def get_budget():
        return Chat.budget()

    
    def query(self, sample):
        chat = Chat.create(f"chat TEST {self.session_id}")
        self.session_id += 1
        
        if 'choices' in sample and sample['choices'] != None :
            message = chat.ask(self.prompt_MCQ(sample),
                    instruction=self.instruction_prompt_MCQ)
        else :
            message = chat.ask(self.prompt_QQ(sample),
                    instruction=self.instruction_prompt_QQ)
        return sample['question'], message, chat
    

    def display(self, question, form_question, answer):
        print(f"##########  {self.session_id}  ##########")
        print(question)
        print("##############################")
        print(form_question)
        print("##############################")
        print(answer)
        print("#########################")

    def query_and_display(self, question):
        form_question, answer = self.query(question)
        self.display(question, form_question, answer)
        
    def multi_query(self, samples, nb_gen, already_started = False):
        if already_started :
            generated = load_data(f'generated_{self.technique}_{nb_gen}_save_conf_3_nb_sent_6.json')
            current = len(generated)
        else :
            generated = []
            current = 0
        total_sample = len(samples)
        for i,sample in enumerate(samples[current:]):
            _, message, chat = self.query(sample)
            generated.append(format_json(sample, chat, message))
            if i % 10 == 9 :
                print(f"  Query {current +i+1}/{total_sample}")
                save_(generated, f"generated_{self.technique}_{nb_gen}_save_conf_3_nb_sent_6.json")
                
        
        save_(generated, f"generated_{self.technique}_{nb_gen}_save_conf_3_nb_sent_6.json")
        return generated
        
    



ANSWERS_MATHIEU = lambda sample : ' '.join([f"""{i+1}. "{answer}" """ for i, answer in enumerate(sample['choices'])])
PROMPT_MATHIEU_MCQ = lambda sample : f"""There is a multiple choice question with {len(sample['choices'])} possible solutions. 
        Here is the question : {sample['question']}. 
        Here are the answers : {ANSWERS_MATHIEU(sample)}."""
PROMPT_MATHIEU_QQ = lambda sample : f"""{sample['question']} """
PROMPT_INSTRUCTION_MATHIEU_QQ = """You are a university teacher for Imaging course. You gave homework to your students and now correct it.
For each question, only suppose what is written in the question.
You first explain what is asked in the question.
Then you give definitions, theorems and lemmas that will be usefull to answer the question. 
Finally, answer to the question and say why it is your answer"""
PROMPT_INSTRUCTION_MATHIEU_MCQ =  """You are a university teacher for Imaging course. You gave homework to your students and now correct it.
For each question, only suppose what is written in the question.
You first explain what is asked in the question.
Then you give definitions, theorems and lemmas that will be usefull to answer the question. 
Next, develop each possible answers saying if it is true or false and why it is the case by giving an different argument for each answers. For each answer structure it by using the following pattern "Answer #x : <the xth answer> : <True or False>. The reason why you answered this" 
An answer is true if we can not find a counter example. An answer is false if we can find a counter example that show that the statement is false. If the answer is false provide a counter-example.
Finally give the number of the final answer (the is unique except if in the question it is said that there might be multiple answers) that is true in any cases."""

PROMPT_MEHRON_QQ = lambda sample : f"""
You are taking an exam for a university course. You will be given a question and your task is to answer correctly to the given question. 

To answer the question, try to think step-by-step and write clear explanations for your answers. Explain what are the important informations needed in order to answer correctly the problem like theorems, general information about terms in the question and formulas.
 
Use the following format:
Question:
```
question here
```
Important informations:
```
Here are the important informations needed in order to answer correctly the problem.
```
Correct Answer:
```
your answer here
```
Explanation:
```
here is an explanation and justification of your answer. 
```
Question:
```
{sample['question']}
``` 
Correct Answer:
"""
PROMPT_MEHRON_MCQ = lambda sample : f"""
You are taking an exam for a university course. You will be given a multiple-choice question and your task is to select only one correct answer from the offered list of options. 

To solve the problem, follow these steps:
- Explain what are the important informations needed in order to answer correctly the problem like theorems, general information about terms in the question and formulas.
- Go step by step through all possible answers from the given list of options.
- For each option, try to think whether this answer is correct or not and explain your reasoning. Is this a complete and appropriate answer to the question asked? 
- When you have checked all the answers from the given list of options, choose the most complete and appropriate answer among them. Do not choose the correct answer until you have checked all the answers from the given list.

Please, for each option from the offered list write an explanation of why you think it's false or true.
 
Use the following format:
Question:
```
question here
```
Choices:
```
here is a list of possible answers
```
Important informations:
```
Here are the important informations needed in order to answer correctly the problem.
```

Correct Answer:
``` Here is the correct answer that you chose from the offered list. format:
variant number. answer ```

Explanation:
```
Here is an explanation and justification for each option why it's true or false. format:
variant number. "TRUE" or "FALSE": Your explanation
```

Question:
```
{sample['question']}
``` 
Choices:
```
{sample['choices']}
```
"""
PROMPT_INSTRUCTION_ANTOINE = """
            You are a teacher in an Institure of technology. Your goal is to provide correct and well explained answer so that your
            student can learn from your explaination. 

            Start by stating the definitions, lemmas, theorems, and formulas that are relevant to the resolution of the exercise.

            You'll have a question and multiple proposed answers. You will list all the answers in the order they were proposed. 
            For each of them you will you will proposed a reasoning justified by definitions and properties

            In a multiple choice question, a propostion is considered true when we can't find any counter example. It has to be true all the time.

            You will finish by simply stating what the answer was.

            Here's examples of interactions :

            Q : 
                There is a multiple choice question with 4 possible solutions. Here is the question : What is the stationary distribution of a Markov chain?. Here is the answers : 1. "The distribution of the initial state of the chain"  2. "The distribution of the final state of the chain"  3. "The distribution that the chain converges to as the number of steps approaches infinity"  4. "The distribution that the chain starts with and maintains throughout its evolution"  
            
            A : 
                To understand the concept of stationary distribution of a Markov chain, we need to first understand what a Markov chain is. A Markov chain is a mathematical model that describes a system that transitions between different states over time. The transition from one state to another is based only on the current state of the system and not on any previous states.

                The stationary distribution of a Markov chain is the distribution that the chain converges to as the number of steps approaches infinity. This means that no matter what the initial state of the chain is, after a large number of steps, the probability of being in each state will approach a fixed distribution, which is the stationary distribution.

                1. "The distribution of the initial state of the chain": False. The distribution of the initial state of the chain is simply the probability distribution of starting in each state. It does not take into account the transitions between states.

                2. "The distribution of the final state of the chain": False. The distribution of the final state of the chain depends on the initial state and the transition probabilities. It is not fixed and does not represent the stationary distribution.

                3. "The distribution that the chain converges to as the number of steps approaches infinity": True. As explained above, the stationary distribution is the distribution that the chain converges to as the number of steps approaches infinity.

                4. "The distribution that the chain starts with and maintains throughout its evolution": Partially True. The stationary distribution is the distribution that the chain approaches after a large number of steps. It is nevertheless possible for a markov chain to start at the stationnary distribution but it's not the usual case.

                Therefore, the correct answer is 3. "The distribution that the chain converges to as the number of steps approaches infinity".

                
            Q :
                There is a multiple choice question with 4 possible solutions. Here is the question : What is the voltage drop through a resistor of $1M\Omega$ of a current with intensity 15 A. Here is the answers : 1. "10V"  2. "100 000 V"  3. "10 000 000 V"  4. "15 000 000 V"  
            A :
                A resistor is an electrical component that limits the flow of current in a circuit. The voltage drop across a resistor is given by Ohm's law: V = IR, where V is the voltage drop, I is the current, and R is the resistance.

                In this case, the resistor has a resistance of $1M\Omega$ and the current has an intensity of 15 A. Therefore, the voltage drop across the resistor is:

                V = IR = 15 A * $1M\Omega$ = 15,000,000 V

                Answer 4, "15,000,000 V" is the correct answer.

            
            Q : 
                There is a multiple choice question with 4 possible solutions. Here is the question : What is the fourier transform of $$f(x) =eta e^{- \alpha x} u(t)$$. Here is the answers : 1. "\frac{\beta}{\alpha + i \omega}"  2. "\frac{\alpha}{\beta + i \omega}"  3. "\frac{\alpha}{\beta - i \omega}"  4. "The fourier transform doesn't exist"  
            A :
                To find the Fourier transform of the given function, we can start by applying the definition of Fourier transform, which is:

                $$F(\omega) = \int_{-\infty}^{\infty} f(t)e^{-i\omega t} dt$$

                Here, we have $f(t) = \beta e^{-\alpha t}u(t)$, where $\beta$ and $\alpha$ are constants, and $u(t)$ is the unit step function.

                First, we can simplify $f(t)$ using the fact that $u(t) = 0$ for $t<0$ and $u(t) = 1$ for $t\geq 0$. Then, we have:

                $$f(t) = \beta e^{-\alpha t}u(t) = 
                \begin{cases}
                0, & t<0 \\
                \beta e^{-\alpha t}, & t\geq 0
                \end{cases}$$

                Now, we can substitute this expression into the definition of Fourier transform and solve the integral. We have:

                $$F(\omega) = \int_{-\infty}^{\infty} f(t)e^{-i\omega t} dt = \int_{0}^{\infty} \beta e^{-\alpha t} e^{-i\omega t} dt$$

                We can simplify this integral by factoring out $\beta$ and using the fact that $e^{a+b} = e^a e^b$. Then, we have:

                $$F(\omega) = \beta \int_{0}^{\infty} e^{-(\alpha+i\omega) t} dt = \beta \left[\frac{-1}{\alpha+i\omega} e^{-(\alpha+i\omega) t}\right]_{0}^{\infty}$$

                Since $e^{-\infty} = 0$, we can simplify this expression as:

                $$F(\omega) = \frac{\beta}{\alpha+i\omega}$$

                Therefore, the correct answer is 1. $\frac{\beta}{\alpha + i \omega}$.

            Q :
                There is a multiple choice question with 4 possible solutions. Here is the question : What is the mathematical equation for the Fourier transform of a continuous signal?. Here is the answers : 1. "$$F(u) = \int f(t) e^{-2πiut} dt$$"  2. "$$F(t) = \int f(u) e^{−2πitu} du$$"  3. "$$F(u) = \sum f(t) e^{−2πiut}$$"  4. "$$F(t) = \sum f(u) e^{−2πitu}$$"  
            A :
                To answer this question, we need to know the mathematical equation for the Fourier transform of a continuous signal. The Fourier transform is a mathematical tool used to decompose a signal into its frequency components.

                1. "$$F(u) = \int f(t) e^{-2πiut} dt$$" : True. This is the correct mathematical equation for the Fourier transform of a continuous signal. It represents the integral of the signal multiplied by a complex exponential.

                2. "$$F(t) = \int f(u) e^{−2πitu} du$$" : False. This equation is not the correct mathematical equation for the Fourier transform of a continuous signal. The variable "t" and "u" are swapped, and the complex exponential has the opposite sign in the exponent.

                3. "$$F(u) = \sum f(t) e^{−2πiut}$$" : False. This equation represents the Fourier series of a periodic signal, not the Fourier transform of a continuous signal. The "sum" symbol implies that the signal is periodic and can be decomposed into a series of complex exponentials.

                4. "$$F(t) = \sum f(u) e^{−2πitu}$$" : False. This equation is not the correct mathematical equation for the Fourier transform of a continuous signal. The variable "t" and "u" are swapped, and the complex exponential has the opposite sign in the exponent.

                Therefore, the correct answer is 1. "$$F(u) = \int f(t) e^{-2πiut} dt$$".

            Q :
                There is a multiple choice question with 4 possible solutions. Here is the question : You have two continuous time signals that are of exponential order , how can you compare their internal frequencies ?. Here is the answers : 1. "the signals known values"  2. "the Z transform"  3. "their derivatives"  4. "their Laplace transform"  
            A :
                The two studied signals are unbounded. In signal processing, an unbounded signal is a signal that grows without limit as time increases, or a signal that has an infinite energy or power. They also are said to be of exponential order. A signal $f(t)$ is said to be of exponential order if there exist positive constants $M$ and $a$ such that $|f(t)| \leq Me^{at}$ for all $t \geq 0$. In other words, the signal grows no faster than an exponential function as $t$ increases.
                To compare the internal frequencies of two continuous time signals that are of exponential order, we need to understand the properties of these signals and the mathematical tools that can be used to analyze them.

                1. "The signals known values": False. The values of the signals alone do not provide enough information to compare their internal frequencies. The signals could have different initial values or amplitudes, but still have the same internal frequency.

                2. "The Z transform": False. The Z transform is a mathematical tool used to analyze discrete time signals, not continuous time signals. It is not applicable in this case.

                3. "Their derivatives": False. The derivatives of the signals can provide information about the rate of change of the signals, but they do not directly provide information about their internal frequencies.

                4. "Their Laplace transform": True. The Laplace transform is a mathematical tool used to analyze continuous time signals, and it can provide information about the internal frequencies of exponential order signals. The Laplace transform of an exponential order signal is a rational function, and the poles of this function correspond to the internal frequencies of the signal.
            """        
# Base conversation agent interface
class BaseConversationAgent:
    def __init__(self, name: str, instructions: str):
        self._name = name
        self._instructions = instructions

    @property
    def name(self) -> str:
        return self._name

    def get_conv(self) -> str:
        raise NotImplementedError("Must implement get_conv")

    def reply(self, question: str) -> str:
        raise NotImplementedError("Must implement reply")

    def reset_chat(self):
        raise NotImplementedError("Must implement reset_chat")

# Custom implementation
class GuidanceConversationAgent(BaseConversationAgent):
    def __init__(self, chat_model, name: str, instructions: str, context_turns: int = 2, temperature=0.1):
        super().__init__(name, instructions)
        self._chat_model = chat_model
        self._my_turns = []
        self._interlocutor_turns = []
        self._went_first = False
        self._context_turns = context_turns
        self.temperature = temperature

    def get_conv(self) -> str:
        return self._my_turns

    def reset_chat(self):
        self._my_turns = []
        self._interlocutor_turns = []
        self._went_first = False

    def reply(self, interlocutor_reply: str | None = None) -> str:
        if interlocutor_reply is None:
            self.reset_chat()
            self._went_first = True
        else:
            self._interlocutor_turns.append(interlocutor_reply)

        my_hist = self._my_turns[(1-self._context_turns):]
        interlocutor_hist = self._interlocutor_turns[-self._context_turns:]

        curr_model = self._chat_model
        with system():
            curr_model += f"{self._instructions}"

        for i in range(len(my_hist)):
            with user():
                curr_model += interlocutor_hist[i]
            with assistant():
                curr_model += my_hist[i]

        if len(interlocutor_hist) > 0:
            with user():
                curr_model += interlocutor_hist[-1]

        with assistant():
            curr_model += gen(name='response', max_tokens=1024, temperature=self.temperature)

        self._my_turns.append(curr_model['response'])
        return curr_model['response']

# LlamaIndex implementation
class LlamaIndexConversationAgent(BaseConversationAgent):
    def __init__(self, llm, name: str, instructions: str, index=None, 
                 chat_token_limit=2500, verbose=False, chat_mode="context"):
        super().__init__(name, instructions)
        self._source_nodes = []
        
        if index is not None:
            self.chat_engine = get_chat_engine_from_index_LlamaIndex(
                llm, index, 
                chat_token_limit=chat_token_limit,
                verbose=verbose,
                chat_mode=chat_mode,
                system_prompt=f'You are a chatbot, able to have normal interactions, as well as talk about data provided.\n\n{self._instructions}'
            )
        else:
            self.chat_engine = SimpleChatEngine.from_defaults(
                llm=llm, 
                system_prompt=self._instructions
            )

    def get_conv(self) -> str:
        return self.chat_engine.chat_history

    def get_source_nodes(self) -> list:
        return self._source_nodes

    def reset_chat(self):
        self.chat_engine.reset()

    def reply(self, question: str) -> tuple[str, any]:
        response = self.chat_engine.stream_chat(question)
        for token in response.response_gen:
            print(token, end="")

        self._source_nodes.append(response.source_nodes)
        return response.response, response

# Unified conversation simulator
def conversation_simulator(
    agent: BaseConversationAgent,
    question_agent: BaseConversationAgent,
    initial_question: str,
    total_turns: int = 5,
    delete_last_question: bool = True,
    only_last: bool = True,
    marker_ch: str = '>>> ',
) -> tuple[list[dict], str]:
    """
    Simulates a conversation between two agents
    
    Args:
        agent: The main conversation agent
        question_agent: Agent that generates questions
        initial_question: Starting question
        total_turns: Number of conversation turns
        delete_last_question: Whether to remove the last unanswered question
        only_last: Whether to only consider the last exchange for context
        marker_ch: Marker character for formatting
    
    Returns:
        Tuple containing conversation turns and formatted text
    """
    conversation_turns = []
    current_question = initial_question
    
    for _ in range(total_turns):
        # Get response from main agent
        response = agent.reply(current_question)
        conversation_turns.append({
            "name": agent.name,
            "text": response
        })
        
        # Generate next question
        if only_last:
            context = f"Consider this question and response.\n\nQuestion: {current_question}\n\nResponse: {response}"
        else:
            context = get_formatted_conversation(initial_question, conversation_turns, marker_ch)
            
        current_question = question_agent.reply(context)
        conversation_turns.append({
            "name": question_agent.name,
            "text": current_question
        })

    if delete_last_question:
        conversation_turns.pop()
        
    formatted_text = format_conversation(initial_question, conversation_turns, marker_ch)
    
    return conversation_turns, formatted_text
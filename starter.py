import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams

# Global variables
page_events = 0

import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import ToolResult, Workbench

playwright_server_params = SseServerParams(
    url="http://localhost:8931/sse",
)

import json
import string
import uuid
from typing import List

import openai
from autogen_core import (
    DefaultTopicId,
    FunctionCall,
    Image,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
    
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from IPython.display import display  # type: ignore
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown


class GroupChatMessage(BaseModel):
    body: UserMessage


class RequestToSpeak(BaseModel):
    pass




class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: OpenAIChatCompletionClient,
        system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        # print(f"\n{'-'*80}\n{self.id.type}:", flush=True)
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        ## generates another message
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content))
        # print(completion.content, flush=True)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=completion.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )



class WriterAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""You are a Writer. 
            You produce A summary and overview of what the web researcher gathers 
            from the internet given what the user wanted.""",
        )


class EditorAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""Your role is to keep all of the other agents on task. 
            Keep the web agent gathering information.
            Make sure that the writer agent is summarizing the findings that the web researcher is reporting and not leaving out any details."""
        )



class UserAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        # When integrating with a frontend, this is where group chat message would be sent to the frontend.
        Console().print(Markdown(f"### {self.id.type}: "))
        Console().print(Markdown(message.body.content))

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        global page_events
        page_events += 1
        if page_events > 20:
            user_input = "Please summarize your findings so far, then type 'exit' to end the conversation."
        user_input =  "Please continue to gather as much information as possible."
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )


class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        # If the message is an approval message from the user, stop the chat.
        if message.body.source == "User":
            assert isinstance(message.body.content, str)
            if message.body.content.lower().strip(string.punctuation).endswith("approve"):
                return
        # Format message history.
        messages: List[str] = []
        for msg in self._chat_history:
            if isinstance(msg.content, str):
                messages.append(f"{msg.source}: {msg.content}")
            elif isinstance(msg.content, list):
                line: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        line.append(item)
                    else:
                        line.append("[Image]")
                messages.append(f"{msg.source}: {', '.join(line)}")
        history = "\n".join(messages)
        # Format roles.
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types, self._participant_descriptions, strict=True
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        # Select the next role to play.
        selector_prompt = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
        system_message = SystemMessage(
            content=selector_prompt.format(
                roles=roles,
                history=history,
                participants=str(
                    [
                        topic_type
                        for topic_type in self._participant_topic_types
                        if topic_type != self._previous_participant_topic_type
                    ]
                ),
            )
        )
        completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)
        assert isinstance(completion.content, str)
        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
                return
        raise ValueError(f"Invalid role selected: {completion.content}")

@dataclass
class Message:
    content: str


class WorkbenchAgent(RoutedAgent):
    def __init__(
        self, model_client: ChatCompletionClient, model_context: ChatCompletionContext, workbench: Workbench
    ) -> None:
        super().__init__("An agent with a workbench")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="""You are a helpful AI assistant with web browsing capabilities.
        You can search the internet, open websites, and interact with web content to find information.
        When asked to research something, use your web browsing tools to gather accurate information.
        Be thorough in your research and provide specific details from reliable sources.""")]
        self._model_client = model_client
        self._model_context = model_context
        self._workbench = workbench
        self._group_chat_topic_type = "group_chat"
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Add the user message to the model context.
        await self._model_context.add_message(UserMessage(content=message.content, source="user"))
        print("---------User Message-----------")
        print(message.content)

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=(await self._workbench.list_tools()),
            cancellation_token=ctx.cancellation_token,
        )

        # Run tool call loop.
        while isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        ):
            print("---------Function Calls-----------")
            for call in create_result.content:
                print(call)

            # Add the function calls to the model context.
            await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))

            # Call the tools using the workbench.
            print("---------Function Call Results-----------")
            results: List[ToolResult] = []
            for call in create_result.content:
                result = await self._workbench.call_tool(
                    call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                )
                results.append(result)
                print(result)

            # Add the function execution results to the model context.
            await self._model_context.add_message(
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result.to_text(),
                            is_error=result.is_error,
                            name=result.name,
                        )
                        for call, result in zip(create_result.content, results, strict=False)
                    ]
                )
            )

            # Run the chat completion again to reflect on the history and function execution results.
            create_result = await self._model_client.create(
                messages=self._system_messages + (await self._model_context.get_messages()),
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )

        # Now we have a single message as the result.
        assert isinstance(create_result.content, str)

        print("---------Final Response-----------")
        print(create_result.content)

        # Add the assistant message to the model context.
        await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))

        # Return the result as a message.
        return Message(content=create_result.content)
        
    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        """Handle group chat messages by storing them in chat history."""
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        Console().print(Markdown(f"### WebAgent received message from {message.body.source}: "))
        Console().print(Markdown(message.body.content))

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        """Handle request to speak in the group chat by generating a response."""
        Console().print(Markdown("### WebAgent: "))
        
        # Prepare system context based on chat history
        context = "Based on the previous conversation, gather information from the web to help with the task."
        
        # Add messages to context
        await self._model_context.add_message(SystemMessage(content=context))
        for msg in self._chat_history:
            if isinstance(msg.content, str):
                await self._model_context.add_message(UserMessage(content=f"{msg.source}: {msg.content}", source="chat_history"))
        
        # Run the chat completion with the tools
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=(await self._workbench.list_tools()),
            cancellation_token=ctx.cancellation_token,
        )

        # Run tool call loop for web research
        while isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        ):
            Console().print(Markdown("**Researching on the web...**"))
            
            # Add the function calls to the model context
            await self._model_context.add_message(AssistantMessage(content=create_result.content, source="WebAgent"))

            # Call the tools using the workbench
            results: List[ToolResult] = []
            for call in create_result.content:
                result = await self._workbench.call_tool(
                    call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                )
                results.append(result)
                Console().print(Markdown(f"**Tool result:** {result.to_text()[:100]}..."))

            # Add the function execution results to the model context
            await self._model_context.add_message(
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result.to_text(),
                            is_error=result.is_error,
                            name=result.name,
                        )
                        for call, result in zip(create_result.content, results, strict=False)
                    ]
                )
            )

            # Run the chat completion again
            create_result = await self._model_client.create(
                messages=self._system_messages + (await self._model_context.get_messages()),
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )

        # Generate final response
        response_content = create_result.content if isinstance(create_result.content, str) else "Error: Unable to generate response"
        Console().print(Markdown(response_content))
        
        # Add to history and publish message
        self._chat_history.append(AssistantMessage(content=response_content, source="WebAgent"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=response_content, source="WebAgent")),
            DefaultTopicId(type=self._group_chat_topic_type),
        )


async def group_chat():
    # Create the model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Set up the MCP workbench for handling browser automation through Playwright
    async with McpWorkbench(playwright_server_params) as workbench:
        # Create a single-threaded agent runtime
        runtime = SingleThreadedAgentRuntime()

        # Set up topic types
        web_agent_topic_type = "WebAgent"
        writer_topic_type = "Writer"
        editor_topic_type = "Editor"
        user_topic_type = "User"
        group_chat_topic_type = "group_chat"
        
        # Set agent descriptions
        web_agent_description = "Web browsing agent capable of searching and interacting with websites"
        writer_description = "Writer for creating content based on research"
        editor_description = "Editor for keeping the team on track"
        user_description = "User for providing guidance and receiving final results."

        # Register the web agent with the runtime using the WorkbenchAgent we defined
        web_agent_type = await WorkbenchAgent.register(
            runtime=runtime,
            type=web_agent_topic_type,
            factory=lambda: WorkbenchAgent(
                model_client=model_client,
                model_context=BufferedChatCompletionContext(buffer_size=10),
                workbench=workbench,
            ),
        )
        
        # Register writer agent
        writer_agent_type = await WriterAgent.register(
            runtime,
            writer_topic_type,
            lambda: WriterAgent(
                description=writer_description,
                group_chat_topic_type=group_chat_topic_type,
                model_client=model_client,
            ),
        )
        
        # Register editor agent
        editor_agent_type = await EditorAgent.register(
            runtime,
            editor_topic_type,
            lambda: EditorAgent(
                description=editor_description,
                group_chat_topic_type=group_chat_topic_type,
                model_client=model_client,
            ),
        )
        
        # Initialize the user agent
        user_agent_type = await UserAgent.register(
            runtime,
            user_topic_type,
            lambda: UserAgent(description=user_description, group_chat_topic_type=group_chat_topic_type),
        )
        
        # Set up topic subscriptions for all agents
        # Web agent subscriptions
        await runtime.add_subscription(TypeSubscription(topic_type=web_agent_topic_type, agent_type=web_agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=web_agent_type.type))
        
        # Writer agent subscriptions
        await runtime.add_subscription(TypeSubscription(topic_type=writer_topic_type, agent_type=writer_agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=writer_agent_type.type))
        
        # Editor agent subscriptions
        await runtime.add_subscription(TypeSubscription(topic_type=editor_topic_type, agent_type=editor_agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=editor_agent_type.type))
        
        # User agent subscriptions
        await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=user_agent_type.type))
        
        # Register group chat manager with the runtime
        group_chat_manager_type = await GroupChatManager.register(
            runtime,
            "group_chat_manager",
            lambda: GroupChatManager(
                participant_topic_types=[web_agent_topic_type, writer_topic_type, editor_topic_type, user_topic_type],
                model_client=model_client,
                participant_descriptions=[
                    web_agent_description,
                    writer_description,
                    editor_description,
                    user_description
                ],
            ),
        )
        await runtime.add_subscription(
            TypeSubscription(topic_type=group_chat_topic_type, agent_type=group_chat_manager_type.type)
        )
        
        # Start the runtime
        runtime.start()
        
        # Create a session ID for this conversation
        session_id = str(uuid.uuid4())
        
        print("\n=== Starting Group Chat ===")
        print("Task: Find info about the bsides conference in 2025 and write an article about it.")
        print("Agents: WebAgent, Writer, Editor, and User")
        print("=== Conversation Start ===\n")
        
        # Publish the initial message to start the group chat
        await runtime.publish_message(
            GroupChatMessage(
                body=UserMessage(
                    content="Find info about the bsides conference in 2025 and write an article about it.",
                    source="User",
                )
            ),
            TopicId(type=group_chat_topic_type, source=session_id),
        )
        
        # Add an explicit message to the group chat manager to trigger the first selection
        # This ensures the conversation gets started
        await runtime.publish_message(
            GroupChatMessage(
                body=UserMessage(
                    content="Please select the first agent to respond to this task.",
                    source="System"
                )
            ),
            DefaultTopicId(type=group_chat_topic_type),
        )
        
        # Wait for the runtime to complete with a timeout to ensure it doesn't run indefinitely
        try:
            global page_events
            page_events = 0  # Reset counter for this session
            
            # Instead of stop_when_idle which might exit too soon, we'll use a sleep
            # and explicitly manage the conversation
            conversation_active = True
            
            # Give the conversation time to develop
            while conversation_active and page_events < 25:  # Use page_events counter as a limiting factor
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # This could be replaced with a more sophisticated check
                if page_events > 0 and page_events % 5 == 0:
                    print(f"\n=== Conversation progress: {page_events} turns ===")
                
                if page_events > 20:
                    # Explicitly request a summary from the writer agent
                    await runtime.publish_message(
                        GroupChatMessage(
                            body=UserMessage(
                                content="Please summarize your findings so far, and then type 'exit' to end the conversation.",
                                source="User"
                            )
                        ),
                        DefaultTopicId(type=group_chat_topic_type),
                    )
                    conversation_active = False
                
            print("\n=== Conversation Complete ===")
        finally:
            # Make sure to properly clean up resources
            await model_client.close()

# Run the group chat
asyncio.run(group_chat())


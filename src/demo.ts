import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { docUserstoryOE1 } from "./data/oe-userstory-1";
import {
  AIMessage,
  HumanMessage,
  type MessageContent,
} from "@langchain/core/messages";

const model = new ChatOpenAI({
  model: "gpt-4.1",
  temperature: 0.7,
});

const template = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user question based on the following context: {context}.",
  ],
  [
    "system",
    `As a QA tester, I have solid expertise in testing web applications and websites, ensuring their functionality, usability, and performance.`,
  ],
  [
    "system",
    `You are a careful assistant helping answer user questions. Use only the information in the retrieved context below. If an answer is not explicitly stated, reply: "The provided information does not contain enough detail to answer that." Do not make up any rules, steps, or details that are not in the given context.`,
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);

const retriever = await initVectorData();

const chain = template.pipe(model);

let beforeHumanMessage = "";
let beforeAIMessage = "";

async function initVectorData() {
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  await vectorStore.addDocuments(docUserstoryOE1);
  const retriever = vectorStore.asRetriever({
    k: 4,
  });
  return retriever;
}

async function chat(question: string): Promise<MessageContent> {
  const retrievedDocs = await retriever.getRelevantDocuments(question);
  const retrievedContent = retrievedDocs.map((doc) => doc.pageContent);

  const res = await chain.invoke({
    input: question,
    chat_history: [
      new HumanMessage(beforeHumanMessage),
      new AIMessage(beforeAIMessage),
    ],
    context: retrievedContent,
  });

  beforeHumanMessage = question;
  beforeAIMessage = String(res.content);

  return res.content;
}

async function main() {
  const prompt = "Type something: ";
  process.stdout.write(prompt);
  for await (const line of console) {
    //     console.log(`You typed: ${line}`);
    const res = await chat(line);
    console.log(`\n\n\n`);
    process.stdout.write(String(res));
    console.log(`\n\n\n`);
    process.stdout.write(prompt);
  }
}

main();

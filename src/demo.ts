import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { data, docUserstoryOE1, text } from "./data/oe-userstory-1";
import {
  AIMessage,
  HumanMessage,
  type MessageContent,
} from "@langchain/core/messages";
import { MarkdownTextSplitter } from "langchain/text_splitter";

const k = 5;

const model = new ChatOpenAI({
  model: "gpt-4.1",
  temperature: 0.7,
});

  const embeddings = new OpenAIEmbeddings();

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
  const vectorStore = new MemoryVectorStore(embeddings);
  // await vectorStore.addDocuments(docUserstoryOE1);
  const chunks = await semanticChunkText(text);
  const documents = chunks.map((chunk) => ({ pageContent: chunk, metadata: {} }));
  await vectorStore.addDocuments(documents);
  const retriever = vectorStore.asRetriever({
    k,
  });
  return retriever;
}

// // Helper: Cosine similarity
// function cosineSimilarity(vecA: number[], vecB: number[]): number {
//   const dot = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
//   const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
//   const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
//   return dot / (normA * normB);
// }

// async function semanticChunkText(text: string, similarity=0.85) {
async function semanticChunkText(text: string) {
  const splitter = new MarkdownTextSplitter();
  const chunks = await splitter.splitText(text);

  return chunks;
  // console.log('chunks', chunks)

  // const vectors = await Promise.all(chunks.map(chunk => embeddings.embedQuery(chunk)));

  // const result: string[] = [];
  // let buffer = chunks[0];

  // for (let i = 1; i < chunks.length; ++i) {
  //   const sim = cosineSimilarity(vectors[i - 1], vectors[i]);
  //   if (sim > similarity) {
  //     buffer += " " + chunks[i];
  //   } else {
  //     result.push(buffer);
  //     buffer = chunks[i];
  //   }
  // }
  // if (buffer) result.push(buffer);
  // return result;
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

// async function testSemanticChunkText() {
//   const semanticChunks = await semanticChunkText(text);
//   console.log(semanticChunks);
// }

// testSemanticChunkText();

main();

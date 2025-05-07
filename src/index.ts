import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0.7,
});

const data = [
  //   "My name is John Doe. I am a software engineer with 5 years of experience in web development. I love coding and learning new technologies.",
  //   "I am Jane Smith, a data scientist with a passion for machine learning. I have worked on various projects involving data analysis and visualization.",
  `Acceptance Criteria of Create Measure Field
1. Admin should be able to define key properties for creating measure : Measurement Name (required) , Filed and column apply the measurement on (required) and conditions.
2. Data source could be measure , table and subquery.
3. Alias should not duplicated.
4. Alias is required except when field type is Table
5. Rule for value type : 
5.1.User able to specify value as either numerical or string value.
6.Rule for Table type :
6.1. User able to select either primary table and joined table. (Note : tables should have been joined before creating field)
6.2. If joined table is removed , the system will prompt a notification indicating that related fields are no longer available.
6.3. User have to GROUP BY data before applying an aggregate function to know how to group the rows together before it can calculate the summary values. Without GROUP BY, the database would not know which rows should be combined or summarized together.

7. Rule for Expression type : 
7.1.User able to apply only numerical column and numerical value in Expression.
8. Rule for Condition type : 
8.1.The value type of True and False is depended on type of selected table. If table is numerical data, user able to specify TRUE/False value only number.`,

  `Acceptance Criteria of Create Measure type Filter
1.Admin able to add multiple query filters.
2.System supports combining multiple filters using AND logic.
3.Admin able to specify the filter key properties : table , column , operation and value.
4.Admin can directly use selected table for creating filter , no need to joined first.
5.If a table or column used in the filter is deleted or unavailable, the system must alert the user and require correction.
6.System should validated all required filed before saving ; otherwise , a proper alert message should display.
7.To save , filter and filed must be created prior saving the measurement `,
];

// const question =
//   "Specify the title name of acceptance criteria that  involve applying math logical and function";

const question = "What data should user have to specify for creating field";
//   "Which user story includes the acceptance criterion that the user unable to input duplicate Alias name and name must be unique.";

async function main() {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const docs = await splitter.createDocuments(data);

  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  await vectorStore.addDocuments(docs);
  //   await vectorStore.addDocuments(
  //     data.map((content) => new Document({ pageContent: content }))
  //   );

  const retriever = vectorStore.asRetriever({
    k: 8,
  });

  console.log("Retrieving documents...");
  const retrievedDocs = await retriever.getRelevantDocuments(question);
  console.log("Retrieved documents:", retrievedDocs);

  const retrievedContent = retrievedDocs.map((doc) => doc.pageContent);
  console.log("Retrieved content:", retrievedContent);

  const template = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user question based on the following context: {context}.",
    ],
    //     ['system', "You are an expert SQL generator. Use the table definitions below to write a SQL query that answers the question."],
    ["user", "{input}"],
  ]);

  const chain = template.pipe(model);
  const res = await chain.invoke({
    input: question,
    context: retrievedContent,
  });

  console.log("---------------------------------------");
  console.log("Question:", question);
  console.log("Response:", res.content);
}

main();

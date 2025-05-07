import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
// import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const model = new ChatOpenAI({
//   model: "gpt-3.5-turbo",
  model: "gpt-4.1",
  temperature: 0.7,
});

// const data = [
//   //   "My name is John Doe. I am a software engineer with 5 years of experience in web development. I love coding and learning new technologies.",
//   //   "I am Jane Smith, a data scientist with a passion for machine learning. I have worked on various projects involving data analysis and visualization.",
//   `Acceptance Criteria for Create Measure Field
// Key Properties Definition:

// Admin must be able to define key properties:
// Measurement Name (required)
// Field and column to apply the measurement on (required)
// Conditions
// Data Source Options:

// Can be a measure, table, or subquery.
// Alias Management:

// Alias must not be duplicated.
// Alias is required unless the field type is Table.
// Value Type Rules:

// User can specify value as either numerical or string.
// Table Type Rules:

// User can select either primary or joined tables (tables must be joined prior).
// If a joined table is removed, the system must notify the user about unavailable fields.
// User must GROUP BY data before applying an aggregate function for summarizing records.`,

//   `Acceptance Criteria for Create Measure Type Filter
// Query Filters:

// Admin can add multiple query filters.
// Filter Logic:

// System supports combining multiple filters with AND logic.
// Filter Key Properties:

// Admin specifies table, column, operation, and value for filters.
// Table Selection:

// Admin can use a selected table for creating a filter without prior joining.
// Error Handling:

// If a table or column in the filter is deleted or unavailable, the system must alert the user and require correction.
// Validation:

// System must validate all required fields before saving; otherwise, display a proper alert message.
// Saving Requirements:

// Filter and field must be created before saving the measurement.`,
// ];

// const data = [
// 	{
// 	  "Create Measure Field": {
// 	    "Key Properties Definition": {
// 	      "Measurement Name": "required",
// 	      "Field and column": "required",
// 	      "Conditions": "optional"
// 	    },
// 	    "Data Source Options": [
// 	      "measure",
// 	      "table",
// 	      "subquery"
// 	    ],
// 	    "Alias Management": {
// 	      "Alias must not be duplicated": true,
// 	      "Alias is required unless field type is Table": true
// 	    },
// 	    "Value Type Rules": {
// 	      "User can specify value": [
// 		"numerical",
// 		"string"
// 	      ]
// 	    },
// 	    "Table Type Rules": {
// 	      "User can select": [
// 		"primary table",
// 		"joined table"
// 	      ],
// 	      "Joined table removal notification": true,
// 	      "GROUP BY requirement": true
// 	    }
// 	  }
// 	},
// 	{
// 	  "Create Measure Type Filter": {
// 	    "Query Filters": {
// 	      "Admin can add multiple filters": true
// 	    },
// 	    "Filter Logic": {
// 	      "AND logic support": true
// 	    },
// 	    "Filter Key Properties": {
// 	      "Table": "required",
// 	      "Column": "required",
// 	      "Operation": "required",
// 	      "Value": "required"
// 	    },
// 	    "Table Selection": {
// 	      "Use selected table without prior joining": true
// 	    },
// 	    "Error Handling": {
// 	      "Alert for deleted or unavailable tables/columns": true
// 	    },
// 	    "Validation": {
// 	      "All required fields validated before saving": true
// 	    },
// 	    "Saving Requirements": {
// 	      "Filter and field creation required before saving": true
// 	    }
// 	  }
// 	}
//       ]

const data = [
	`### Acceptance Criteria for Create Measure Field

1. **Key Properties Definition:**
   - Admin must be able to define key properties:
     - Measurement Name (required)
     - Field and column to apply the measurement on (required)
     - Conditions

2. **Data Source Options:**
   - Can be a measure, table, or subquery.

3. **Alias Management:**
   - Alias must not be duplicated.
   - Alias is required unless the field type is Table.

4. **Value Type Rules:**
   - User can specify value as either numerical or string.

5. **Table Type Rules:**
   - User can select either primary or joined tables (tables must be joined prior).
   - If a joined table is removed, the system must notify the user about unavailable fields.
   - User must GROUP BY data before applying an aggregate function for summarizing records.

### Acceptance Criteria for Create Measure Type Filter

1. **Query Filters:**
   - Admin can add multiple query filters.

2. **Filter Logic:**
   - System supports combining multiple filters with AND logic.

3. **Filter Key Properties:**
   - Admin specifies table, column, operation, and value for filters.

4. **Table Selection:**
   - Admin can use a selected table for creating a filter without prior joining.

5. **Error Handling:**
   - If a table or column in the filter is deleted or unavailable, the system must alert the user and require correction.

6. **Validation:**
   - System must validate all required fields before saving; otherwise, display a proper alert message.

7. **Saving Requirements:**
   - Filter and field must be created before saving the measurement.
`
]

// const question =
//   "Specify the title name of acceptance criteria that  involve applying math logical and function";

const question = "What data should user have to specify for creating field"
//"Can user configure multiple function in a measure such as configure group by and Join together";
//   "Which user story includes the acceptance criterion that the user unable to input duplicate Alias name and name must be unique.";

async function main() {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const docs = await splitter.createDocuments(data);

  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
//   const documents = data.map(
// 	(item) =>
// 	  new Document({
// 		pageContent: JSON.stringify(item),
// 		metadata: {},
// 	  })
//   );
  await vectorStore.addDocuments(docs);
  //   await vectorStore.addDocuments(
  //     data.map((content) => new Document({ pageContent: content }))
  //   );

  const retriever = vectorStore.asRetriever({
    k: 4,
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
    [
	"system",
	`You are a careful assistant helping answer user questions. Use only the information in the retrieved context below. If an answer is not explicitly stated, reply: "The provided information does not contain enough detail to answer that." Do not make up any rules, steps, or details that are not in the given context.`
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

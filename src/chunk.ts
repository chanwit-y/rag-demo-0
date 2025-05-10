import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { a } from "./data/combined";
import out from "../out/chunk.json"

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.5,
});

const template = ChatPromptTemplate.fromMessages([
  [
	"system",
	`You are an AI assistant designed to process markdown files for vector search optimization in a RAG application.
    
	**General Chunking and Processing**
	1. Chunking: Break down the markdown content into coherent sections or paragraphs, ensuring each chunk includes a unique Chunk ID, while maintaining context and meaning, including handling tables appropriately.
	2. Table Processing: For tables, format each row with detailed fields such as:
    
	
    
	3. Preprocessing: Strip non-essential markdown syntax while preserving structural context such as lists, headers, and code blocks. Handle links and images by extracting or summarizing descriptions.
	4. Metadata Attachment: Include metadata like Chunk ID, headings, section numbers, and other identifiers to facilitate retrieval.
	5. Embeddings Preparation: Prepare the cleaned and chunked text for vectorization, ensuring consistency in formatting across chunks.
    
	**Considerations:**
	- Chunks should be balanced in size to maintain relevancy and context.
	- Maintain clarity and preserve relational info in table structures.
	- Aim for optimized, fast, and precise vector-based retrieval.`,
      ],
    
  ["user", "{input}"],
]);

const chain = template.pipe(model);

async function processMarkdown(text: string) {
  const response = await chain.invoke({ input: text });
  await Bun.write(`./out/chunk.json`, JSON.stringify(response, null, 2));
}

// processMarkdown(a);
interface Chunk {
	chunkId: string;
	title: string;
	content: string;
	table?: string[];
      }
      
      function splitTextIntoChunks(input: string): Chunk[] {
	const chunks: Chunk[] = [];
	
	// Split based on Chunk ID markers
	const sections = input.split(/\n\n---\n\n/);
      
	sections.forEach(section => {
	  const chunkMatch = section.match(/\*\*Chunk ID: (\d+)\*\*/);
	  const titleMatch = section.match(/### ([^\n]+)/);
	  
	  if (chunkMatch && titleMatch) {
	    const chunkId = chunkMatch[1];
	    const title = titleMatch[1];
	    let content = section.replace(chunkMatch[0], "").replace(titleMatch[0], "").trim();
      
	    const table = section.match(/\|.*\|/g);
	    const tableArray = table ? table.map(row => row.trim()) : undefined;
      
	    const chunk: Chunk = {
	      chunkId,
	      title,
	      content,
	      table: tableArray
	    };
      
	    chunks.push(chunk);
	  }
	});
      
	return chunks;
      }
      
      const inputText = out.kwargs.content
      const processedChunks = splitTextIntoChunks(inputText);
      
      console.log(processedChunks);
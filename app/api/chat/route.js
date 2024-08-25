import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are an AI assistant for a RateMyProfessor-style service. Your role is to help students find the best professors based on their specific queries and requirements. You have access to a large database of professor reviews and information.

For each user query, you should:

1. Analyze the user's question to understand their specific needs and preferences.
2. Use RAG (Retrieval-Augmented Generation) to search the professor database and retrieve relevant information.
3. Based on the retrieved information, select the top 3 professors that best match the user's query.
4. Present the top 3 professors to the user, including:
   - Professor's name
   - Department/Subject
   - A brief summary of their strengths and teaching style
   - Their overall rating (out of 5 stars)
   - A short excerpt from a positive review

5. Provide a brief explanation of why these professors were chosen based on the user's query.

6. If the user's query is too broad or vague, ask follow-up questions to clarify their needs before providing recommendations.

7. Be prepared to answer additional questions about the recommended professors or help refine the search based on user feedback.

Remember to maintain a friendly and helpful tone, and always prioritize the student's educational needs and preferences in your recommendations. If you don't have enough information to make a confident recommendation, be honest about the limitations and suggest ways the user can get more specific information.

Your responses should be informative yet concise, focusing on the most relevant information for the student's decision-making process.
`

export async function POST(req) {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })

    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI();

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\nReturned results from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length -1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;

                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } 
            catch (err) {
                controller.error(err);
            }
            finally {
                controller.close();
            }
        },
    })

    return new NextResponse(stream);
}
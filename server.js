const express = require('express');
const cors = require('cors');
require('dotenv').config();
const { OpenAI } = require('openai');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Global request logging middleware
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
    if (req.method === 'POST') {
        console.log('Request Body:', JSON.stringify(req.body, null, 2));
    }
    next();
});

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    timeout: 30000, // 30 seconds timeout
});

// Process-level error handling
process.on('uncaughtException', (err) => {
    console.error('UNCAUGHT EXCEPTION:', err);
    // Keep running if possible, or exit cleanly
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('UNHANDLED REJECTION:', reason);
});

app.post('/api/generate-cards', async (req, res) => {
    console.log('Received request to /api/generate-cards');
    try {
        const { topic, context } = req.body;
        console.log('Request body topic:', topic);
        if (context) console.log(`Context provided with ${context.length} existing cards.`);

        if (!topic) {
            console.warn('Topic is missing in request body');
            return res.status(400).json({ error: 'Topic is required' });
        }

        console.log('Checking OpenAI API Key...');
        if (!process.env.OPENAI_API_KEY) {
            console.error('OPENAI_API_KEY is not set in environment variables');
            throw new Error('OPENAI_API_KEY missing');
        }

        const model = process.env.OPENAI_MODEL || "gpt-5.2";
        console.log(`Calling OpenAI API with model: ${model} (Streaming)`);

        let contextString = "";
        if (context && Array.isArray(context) && context.length > 0) {
            contextString = `\n\nIMPORTANT: The student has already studied the following concepts. Do NOT generate cards for these exact concepts again. Instead, focus on related but new concepts, advanced details, or different aspects of the topic:\n${context.join(', ')}`;
        }

        // Set headers for streaming
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Transfer-Encoding', 'chunked');

        const stream = await openai.chat.completions.create({
            model: model,
            messages: [
                {
                    role: "system",
                    content: `You are an expert tutor used for advanced technical topics. Create 10 educational flashcards to help a student learn the requested topic in depth.
                    
                    For each card:
                    1. "front": A clear, thought-provoking question or concept name.
                    2. "back": A comprehensive explanation (3-6 sentences) that fully answers the question. Be detailed and pedagogical. IMPORTANT: Break your explanation into logical paragraphs using newline characters (\\n) to separate distinct ideas or points. This makes the content more readable and easier to digest.
                    3. "code": (Highly Recommended) If the topic involves ANY programming, mathematics, commands, configuration, or technical syntax, YOU MUST PROVIDE A RELEVANT CODE SNIPPET. If it is a non-technical topic, you may leave it empty. But lean towards providing concrete examples/code whenever possible.

                    IMPORTANT: You must stream the response as Newline Delimited JSON (NDJSON).
                    Each line must be a valid, standalone JSON object representing ONE flashcard.
                    Do not wrap the result in a list or an object "flashcards".
                    Do not return markdown formatting (like \`\`\`json).
                    Just one JSON object per line.
                    
                    Example output format:
                    {"front": "Question", "back": "Detailed answer...", "code": "const x = 1;"}
                    
                    ${contextString}`
                },
                {
                    role: "user",
                    content: `Generate flashcards for the topic: ${topic}`
                }
            ],
            stream: true,
        });

        console.log('OpenAI stream started');

        let buffer = "";

        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || "";
            buffer += content;

            // Process buffer for newlines to check for complete JSON objects
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);

                if (line) {
                    try {
                        // Check if it's potentially valid JSON (starts with {)
                        // This handles cases where the model might output extra text or markdown code blocks occasionally
                        const cleanedLine = line.replace(/^```json\s*/, '').replace(/\s*```$/, '');
                        if (cleanedLine.startsWith('{')) {
                            // Verify it parses before sending
                            JSON.parse(cleanedLine);
                            res.write(cleanedLine + "\n");
                        }
                    } catch (e) {
                        // Incomplete or invalid JSON line, might be part of a larger object usually not if prompts obeyed 
                        // or just noise. For now, we assume strict one-line JSONs due to prompt.
                        console.warn('Skipping invalid JSON line:', line);
                    }
                }
            }
        }

        // Handle any remaining buffer
        if (buffer.trim()) {
            try {
                const cleanedLine = buffer.trim().replace(/^```json\s*/, '').replace(/\s*```$/, '');
                if (cleanedLine.startsWith('{')) {
                    JSON.parse(cleanedLine);
                    res.write(cleanedLine + "\n");
                }
            } catch (e) {
                console.warn('Final buffer content was not valid JSON:', buffer);
            }
        }

        res.end();
        console.log('Stream completed');

    } catch (error) {
        console.error('CRITICAL ERROR in /api/generate-cards:', error);
        // If headers haven't been sent, send error JSON
        if (!res.headersSent) {
            res.status(500).json({ error: 'Failed to generate flashcards', details: error.message });
        } else {
            // Stream was already open, just end it; client will handle truncated stream
            res.end();
        }
    }
});

const server = app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

server.on('error', (e) => {
    if (e.code === 'EADDRINUSE') {
        console.error(`ERROR: Port ${port} is already in use!`);
        console.error(`Please kill the process running on port ${port} or check your .env file.`);
        process.exit(1);
    } else {
        console.error('Server error:', e);
    }
});

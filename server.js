const express = require('express');
const cors = require('cors');
require('dotenv').config();
const { OpenAI } = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const winston = require('winston');

// Configure Winston logger
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp({
            format: 'YYYY-MM-DD HH:mm:ss'
        }),
        winston.format.errors({ stack: true }),
        winston.format.printf(({ timestamp, level, message, stack }) => {
            if (stack) {
                return `[${timestamp}] ${level.toUpperCase()}: ${message}\n${stack}`;
            }
            return `[${timestamp}] ${level.toUpperCase()}: ${message}`;
        })
    ),
    transports: [
        new winston.transports.Console()
    ]
});

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Global request logging middleware
app.use((req, res, next) => {
    logger.info(`${req.method} ${req.originalUrl}`);
    if (req.method === 'POST') {
        logger.debug('Request Body: ' + JSON.stringify(req.body, null, 2));
    }
    next();
});

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    timeout: 30000, // 30 seconds timeout
});

// Process-level error handling
process.on('uncaughtException', (err) => {
    logger.error('UNCAUGHT EXCEPTION:', err);
    // Keep running if possible, or exit cleanly
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('UNHANDLED REJECTION:', reason);
});

app.post('/api/generate-cards', async (req, res) => {
    logger.info('Received request to /api/generate-cards');
    try {
        const { topic, context } = req.body;
        logger.info(`Request body topic: ${topic}`);
        if (context) logger.info(`Context provided with ${context.length} existing cards.`);

        if (!topic) {
            logger.warn('Topic is missing in request body');
            return res.status(400).json({ error: 'Topic is required' });
        }

        let contextString = "";
        if (context && Array.isArray(context) && context.length > 0) {
            contextString = `\n\nIMPORTANT: The student has already studied the following concepts. Do NOT generate cards for these exact concepts again. Instead, focus on related but new concepts, advanced details, or different aspects of the topic:\n${context.join(', ')}`;
        }

        const provider = process.env.AI_PROVIDER || 'openai';
        logger.info(`Selected AI Provider: ${provider}`);

        // Set headers for streaming
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Transfer-Encoding', 'chunked');

        let stream;
        let iterator;
        let providerUsage = { inputTokens: 0, outputTokens: 0 };

        if (provider === 'gemini') {
            logger.debug('Checking Gemini API Key...');
            if (!process.env.GEMINI_API_KEY) {
                // Check for fallback to OpenAI key if user tries to use Gemini but only has OpenAI key? No, easier to fail.
                logger.error('GEMINI_API_KEY is not set');
                throw new Error('GEMINI_API_KEY missing');
            }

            const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
            const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash";
            logger.info(`Calling Gemini API with model: ${modelName}`);

            const model = genAI.getGenerativeModel({
                model: modelName,
                systemInstruction: `You are an expert tutor used for advanced technical topics. Create 10 educational flashcards to help a student learn the requested topic in depth.
                    
                    Stream response as Newline Delimited JSON (NDJSON).
                    Each line must be a valid, standalone JSON object representing ONE flashcard.
                    Do not return markdown formatting.
                    
                    Required properties per object:
                    - "front": Question/Concept
                    - "back": Explanation (2-4 sentences)
                    - "code": Optional code snippet
                    
                    ${contextString}`
            });

            const result = await model.generateContentStream(`Generate flashcards for the topic: ${topic}`);
            stream = result.stream;

            // Adapt Gemini stream to iterator
            iterator = (async function* () {
                for await (const chunk of stream) {
                    if (chunk.usageMetadata) {
                        providerUsage.inputTokens = chunk.usageMetadata.promptTokenCount;
                        providerUsage.outputTokens = chunk.usageMetadata.candidatesTokenCount;
                    }
                    yield chunk.text();
                }
            })();

        } else {
            // OpenAI Default
            logger.debug('Checking OpenAI API Key...');
            if (!process.env.OPENAI_API_KEY) {
                logger.error('OPENAI_API_KEY is not set');
                throw new Error('OPENAI_API_KEY missing');
            }

            const model = process.env.OPENAI_MODEL || "gpt-5.2";
            logger.info(`Calling OpenAI API with model: ${model}`);

            const openaiStream = await openai.chat.completions.create({
                model: model,
                messages: [
                    {
                        role: "system",
                        content: `You are an expert tutor used for advanced technical topics. Create 10 educational flashcards to help a student learn the requested topic in depth.
                        
                        For each card:
                        1. "front": A clear, thought-provoking question or concept name.
                        2. "back": A clear, concise explanation (2-4 sentences). Focus on the core concept immediately. Break into paragraphs if needed.
                        3. "code": (Conditional) ONLY provide this if the topic is explicitly technical matching programming/math/tools. Otherwise leave empty string "".

                        IMPORTANT: You must stream the response as Newline Delimited JSON (NDJSON).
                        Each line must be a valid, standalone JSON object representing ONE flashcard.
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
                stream_options: { include_usage: true }
            });

            // Adapt OpenAI stream to iterator
            iterator = (async function* () {
                for await (const chunk of openaiStream) {
                    if (chunk.usage) {
                        providerUsage.inputTokens = chunk.usage.prompt_tokens;
                        providerUsage.outputTokens = chunk.usage.completion_tokens;
                    }
                    const content = chunk.choices[0]?.delta?.content || "";
                    if (content) yield content;
                }
            })();
        }

        const apiCallStart = Date.now();
        logger.info('Stream started');

        let buffer = "";
        let firstChunkReceived = false;
        let cardCount = 0;

        for await (const content of iterator) {
            if (!firstChunkReceived) {
                logger.info(`[PERF] First content chunk received at +${Date.now() - apiCallStart}ms`);
                firstChunkReceived = true;
            }

            buffer += content;

            // Process buffer for newlines to check for complete JSON objects
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);

                if (line) {
                    logger.debug(`Processing line length: ${line.length}`);
                    try {
                        const cleanedLine = line.replace(/^```json\s*/, '').replace(/\s*```$/, '');
                        if (cleanedLine.startsWith('{')) {
                            const parsed = JSON.parse(cleanedLine);
                            logger.info(`Successfully parsed card #${++cardCount}`);
                            res.write(cleanedLine + "\n");
                        }
                    } catch (e) {
                        logger.warn('Skipping invalid JSON line/segment: ' + line.substring(0, 50));
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
                    logger.info(`Successfully parsed final card #${++cardCount}`);
                    res.write(cleanedLine + "\n");
                }
            } catch (e) {
                logger.warn('Final buffer content was not valid JSON: ' + buffer.substring(0, 50));
            }
        }

        res.end();
        logger.info('Stream completed');

        // Calculate and Log Cost
        if (providerUsage.inputTokens > 0 || providerUsage.outputTokens > 0) {
            let inputCostPerM = 0;
            let outputCostPerM = 0;

            if (provider === 'gemini') {
                inputCostPerM = parseFloat(process.env.GEMINI_INPUT_COST_PER_MILLION || '0.10');
                outputCostPerM = parseFloat(process.env.GEMINI_OUTPUT_COST_PER_MILLION || '0.40');
            } else {
                inputCostPerM = parseFloat(process.env.OPENAI_INPUT_COST_PER_MILLION || '2.50');
                outputCostPerM = parseFloat(process.env.OPENAI_OUTPUT_COST_PER_MILLION || '10.00');
            }

            const inputCost = (providerUsage.inputTokens / 1000000) * inputCostPerM;
            const outputCost = (providerUsage.outputTokens / 1000000) * outputCostPerM;
            const totalCost = inputCost + outputCost;

            logger.info(`[COST] Provider: ${provider}`);
            logger.info(`[COST] Usage: ${providerUsage.inputTokens} input tokens, ${providerUsage.outputTokens} output tokens`);
            logger.info(`[COST] Estimated Cost: $${totalCost.toFixed(6)}`);
        } else {
            logger.warn('[COST] usage data was not returned/collected correctly.');
        }

    } catch (error) {
        logger.error('CRITICAL ERROR in /api/generate-cards:', error);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Failed to generate flashcards', details: error.message });
        } else {
            res.end();
        }
    }
});

const server = app.listen(port, () => {
    logger.info(`Server running at http://localhost:${port}`);
});

server.on('error', (e) => {
    if (e.code === 'EADDRINUSE') {
        logger.error(`ERROR: Port ${port} is already in use!`);
        logger.error(`Please kill the process running on port ${port} or check your .env file.`);
        process.exit(1);
    } else {
        logger.error('Server error:', e);
    }
});

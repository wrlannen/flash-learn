const express = require('express');
const cors = require('cors');
require('dotenv').config();
const { OpenAI } = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const winston = require('winston');

function createLogger() {
    return winston.createLogger({
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
}

const logger = createLogger();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

function createRequestLoggingMiddleware(loggerInstance) {
    return (req, res, next) => {
        loggerInstance.info(`${req.method} ${req.originalUrl}`);
        if (req.method === 'POST') {
            loggerInstance.debug('Request Body: ' + JSON.stringify(req.body, null, 2));
        }
        next();
    };
}

app.use(createRequestLoggingMiddleware(logger));

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    timeout: 30000, // 30 seconds timeout
});

function buildStudiedConceptsContextString(context) {
    if (!context || !Array.isArray(context) || context.length === 0) return "";

    return `\n\nIMPORTANT: The student has already studied the following concepts. Do NOT generate cards for these exact concepts again. Instead, focus on related but new concepts, advanced details, or different aspects of the topic:\n${context.join(', ')}`;
}

function getAiProvider() {
    return process.env.AI_PROVIDER || 'openai';
}

function setStreamingHeaders(res) {
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Transfer-Encoding', 'chunked');
}

function cleanPossibleMarkdownJsonFence(text) {
    return text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
}

function parseAndWriteNdjsonLine({ line, res, loggerInstance, cardCountRef }) {
    const cleanedLine = cleanPossibleMarkdownJsonFence(line);
    if (!cleanedLine.startsWith('{')) return;

    JSON.parse(cleanedLine);
    cardCountRef.count += 1;
    loggerInstance.info(`Successfully parsed card #${cardCountRef.count}`);
    res.write(cleanedLine + "\n");
}

async function streamNdjsonFromTextIterator({ iterator, res, loggerInstance }) {
    const apiCallStart = Date.now();
    loggerInstance.info('Stream started');

    let buffer = "";
    let firstChunkReceived = false;
    const cardCountRef = { count: 0 };

    for await (const content of iterator) {
        if (!firstChunkReceived) {
            loggerInstance.info(`[PERF] First content chunk received at +${Date.now() - apiCallStart}ms`);
            firstChunkReceived = true;
        }

        buffer += content;

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + 1);

            if (!line) continue;

            loggerInstance.debug(`Processing line length: ${line.length}`);
            try {
                parseAndWriteNdjsonLine({ line, res, loggerInstance, cardCountRef });
            } catch (e) {
                loggerInstance.warn('Skipping invalid JSON line/segment: ' + line.substring(0, 50));
            }
        }
    }

    if (buffer.trim()) {
        try {
            const line = buffer.trim();
            parseAndWriteNdjsonLine({ line, res, loggerInstance, cardCountRef });
        } catch (e) {
            loggerInstance.warn('Final buffer content was not valid JSON: ' + buffer.substring(0, 50));
        }
    }

    loggerInstance.info('Stream completed');
    return cardCountRef.count;
}

function getCostPerMillionTokens(provider) {
    if (provider === 'gemini') {
        return {
            input: parseFloat(process.env.GEMINI_INPUT_COST_PER_MILLION || '0.10'),
            output: parseFloat(process.env.GEMINI_OUTPUT_COST_PER_MILLION || '0.40'),
        };
    }

    return {
        input: parseFloat(process.env.OPENAI_INPUT_COST_PER_MILLION || '2.50'),
        output: parseFloat(process.env.OPENAI_OUTPUT_COST_PER_MILLION || '10.00'),
    };
}

function logEstimatedCost({ provider, providerUsage, loggerInstance }) {
    if (providerUsage.inputTokens <= 0 && providerUsage.outputTokens <= 0) {
        loggerInstance.warn('[COST] usage data was not returned/collected correctly.');
        return;
    }

    const costPerM = getCostPerMillionTokens(provider);
    const inputCost = (providerUsage.inputTokens / 1000000) * costPerM.input;
    const outputCost = (providerUsage.outputTokens / 1000000) * costPerM.output;
    const totalCost = inputCost + outputCost;

    loggerInstance.info(`[COST] Provider: ${provider}`);
    loggerInstance.info(`[COST] Usage: ${providerUsage.inputTokens} input tokens, ${providerUsage.outputTokens} output tokens`);
    loggerInstance.info(`[COST] Estimated Cost: $${totalCost.toFixed(6)}`);
}

async function createGeminiTextIterator({ topic, contextString, providerUsage, loggerInstance }) {
    loggerInstance.debug('Checking Gemini API Key...');
    if (!process.env.GEMINI_API_KEY) {
        loggerInstance.error('GEMINI_API_KEY is not set');
        throw new Error('GEMINI_API_KEY missing');
    }

    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash";
    loggerInstance.info(`Calling Gemini API with model: ${modelName}`);

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
    const stream = result.stream;

    return (async function* () {
        for await (const chunk of stream) {
            if (chunk.usageMetadata) {
                providerUsage.inputTokens = chunk.usageMetadata.promptTokenCount;
                providerUsage.outputTokens = chunk.usageMetadata.candidatesTokenCount;
            }
            yield chunk.text();
        }
    })();
}

async function createOpenAiTextIterator({ topic, contextString, providerUsage, loggerInstance }) {
    loggerInstance.debug('Checking OpenAI API Key...');
    if (!process.env.OPENAI_API_KEY) {
        loggerInstance.error('OPENAI_API_KEY is not set');
        throw new Error('OPENAI_API_KEY missing');
    }

    const model = process.env.OPENAI_MODEL || "gpt-5.2";
    loggerInstance.info(`Calling OpenAI API with model: ${model}`);

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

    return (async function* () {
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

async function createProviderTextIterator({ provider, topic, contextString, providerUsage, loggerInstance }) {
    if (provider === 'gemini') {
        return createGeminiTextIterator({ topic, contextString, providerUsage, loggerInstance });
    }

    return createOpenAiTextIterator({ topic, contextString, providerUsage, loggerInstance });
}

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

        const contextString = buildStudiedConceptsContextString(context);
        const provider = getAiProvider();
        logger.info(`Selected AI Provider: ${provider}`);

        setStreamingHeaders(res);

        const providerUsage = { inputTokens: 0, outputTokens: 0 };
        const iterator = await createProviderTextIterator({
            provider,
            topic,
            contextString,
            providerUsage,
            loggerInstance: logger,
        });

        await streamNdjsonFromTextIterator({ iterator, res, loggerInstance: logger });
        res.end();

        logEstimatedCost({ provider, providerUsage, loggerInstance: logger });

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

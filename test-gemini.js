require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');

async function testGemini() {
    console.log("Testing Gemini Latency...");
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        console.error("No GEMINI_API_KEY found.");
        return;
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash";
    console.log(`Model: ${modelName}`);

    const model = genAI.getGenerativeModel({ model: modelName });

    const start = Date.now();
    console.log("Sending request...");
    try {
        const result = await model.generateContentStream("Explain quantum computing in 1 sentence.");

        let first = false;
        for await (const chunk of result.stream) {
            if (!first) {
                console.log(`First chunk received after: ${Date.now() - start}ms`);
                first = true;
            }
            process.stdout.write(chunk.text());
        }
        console.log(`\n\nTotal time: ${Date.now() - start}ms`);
    } catch (e) {
        console.error("Error:", e);
    }
}

testGemini();

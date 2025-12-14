# 🧠 FlashLearn

<p align="center">
  <img src="assets/demo-screenshot.png" alt="FlashLearn Demo" width="800">
</p>

An intelligent flashcard generation application powered by advanced AI models (OpenAI or Google Gemini) that creates personalized study materials from any topic. Perfect for students, educators, and lifelong learners looking to enhance their study sessions with AI-generated content.

## 🚀 Quick Start

### Using Docker (Fastest)
```bash
git clone <repository-url>
cd ai-flashcard-generator
cp .env.example .env
# Edit .env with your API keys
make docker-up
```
Visit [http://localhost:3000](http://localhost:3000)

### Using Make
```bash
git clone <repository-url>
cd ai-flashcard-generator
make install
cp .env.example .env
# Edit .env with your API keys
make dev
```
Visit [http://localhost:3000](http://localhost:3000)

### Using npm
```bash
git clone <repository-url>
cd ai-flashcard-generator
npm install
cp .env.example .env
# Edit .env with your API keys
node server.js
```
Visit [http://localhost:3000](http://localhost:3000)

## ✨ Features

- **🤖 Multi-AI Support**: Works with OpenAI GPT models or Google Gemini for intelligent flashcard generation
- **🎯 Context-Aware**: Avoids repeating previously studied concepts when generating new cards
- **🎨 Beautiful Dark Mode UI**: Modern, premium interface built with Tailwind CSS
- **🔄 Interactive Flip Animation**: Smooth card flip effects for an engaging study experience
- **📊 Progress Tracking**: Keep track of how many cards you've studied
- **🔍 Search & Filter**: Quickly find specific flashcards in your collection
- **💾 Local Storage**: Your flashcards are saved locally in your browser
- **⚡ Real-time Streaming**: Fast flashcard creation with streaming responses
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **💰 Cost Tracking**: Monitor API usage and estimated costs

## 🚀 Tech Stack

### Backend
- **Node.js** with **Express.js** - Fast, minimalist web framework
- **OpenAI API** - GPT-4 and other advanced models for intelligent content generation
- **Google Gemini API** - Alternative AI model for content generation
- **Winston Logger** - Comprehensive logging and monitoring
- **CORS** - Cross-origin resource sharing support
- **dotenv** - Environment variable management

### Frontend
- **HTML5** - Semantic markup
- **Vanilla JavaScript** - No framework overhead
- **Tailwind CSS 4.x** - Utility-first CSS framework
- **CSS3** - Custom animations and transitions

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v14 or higher)
- **npm** (Node Package Manager)
- **API Key** - Choose one of:
  - **OpenAI API Key** - [Get one here](https://platform.openai.com/api-keys) (GPT-4 recommended)
  - **Google Gemini API Key** - [Get one here](https://makersuite.google.com/app/apikey) (Optional alternative)

## 🛠️ Installation

### Option 1: Using Make (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-flashcard-generator
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

4. **Configure your `.env` file**
   
   Open `.env` and configure your preferred AI provider:

   **For OpenAI:**
   ```env
   AI_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-5.2
   PORT=3000
   ```

   **For Google Gemini:**
   ```env
   AI_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   PORT=3000
   ```

### Option 2: Using npm

Follow the same steps but replace `make install` with `npm install`.

### Option 3: Using Docker 🐳

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-flashcard-generator
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Build and run with Docker Compose**
   ```bash
   make docker-up
   # Or: docker-compose up -d
   ```

4. **View logs**
   ```bash
   make docker-logs
   # Or: docker-compose logs -f
   ```

## 🏃 Running the Application

### Development (Local)

**Using Make:**
```bash
make dev
```

**Using npm:**
```bash
node server.js
```

### Production (Docker)

```bash
make docker-up
```

### Start generating flashcards!
- Enter a topic you want to learn
- Click "Generate Cards"
- Wait for AI to create your personalized flashcards
- Click on any card to flip and reveal the answer

## 📁 Project Structure

```
ai-flashcard-generator/
├── public/                 # Static frontend files
│   ├── index.html         # Main HTML file with Tailwind CSS
│   ├── script.js          # Frontend JavaScript logic
│   └── favicon.png        # App icon
├── src/                   # Source files (Tailwind config)
│   └── input.css         # Tailwind input styles
├── server.js             # Express server & API endpoints
├── package.json          # Node.js dependencies & scripts
├── tailwind.config.js    # Tailwind CSS configuration
├── .env.example          # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔌 API Endpoints

### `POST /api/generate-cards`

Generates flashcards for a given topic using your configured AI provider (OpenAI or Google Gemini).

**Request Body:**
```json
{
  "topic": "JavaScript Promises",
  "context": ["Promise basics", "async/await"] // Optional: previously studied concepts
}
```

**Response:**
```json
[
  {
    "front": "What is a Promise in JavaScript?",
    "back": "A Promise is an object representing the eventual completion or failure of an asynchronous operation."
  },
  {
    "front": "What are the three states of a Promise?",
    "back": "Pending, Fulfilled, and Rejected."
  }
]
```

**Error Response:**
```json
{
  "error": "Failed to generate flashcards",
  "details": "Error message"
}
```

## 🎯 Usage Tips

1. **Be Specific with Topics**: Instead of "Math", try "Quadratic Equations" for better results
2. **Use Context Feature**: Generate new cards for the same topic to get different perspectives without repetition
3. **Review Regularly**: Use the card counter to track your progress
4. **Search Function**: Use the search bar to quickly find specific cards in large collections

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AI_PROVIDER` | ❌ No | `openai` | 'openai' or 'gemini' |
| `OPENAI_API_KEY` | ✅ if openai | - | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ No | `gpt-5.2` | OpenAI model to use |
| `GEMINI_API_KEY` | ✅ if gemini | - | Your Google Gemini API key |
| `GEMINI_MODEL` | ❌ No | `gemini-2.5-flash` | Gemini model to use |
| `PORT` | ❌ No | `3000` | Server port |

### Tailwind CSS

The project uses Tailwind CSS 4.x. The configuration can be found in `tailwind.config.js`.

## 🐛 Troubleshooting

### Port Already in Use
```
ERROR: Port 3000 is already in use!
```
**Solution**: Change the `PORT` in your `.env` file or kill the process using port 3000.

### OpenAI/Gemini API Key Missing
```
OPENAI_API_KEY missing
# or
GEMINI_API_KEY missing
```

**Solution**: Make sure you have set either `OPENAI_API_KEY` or `GEMINI_API_KEY` in your `.env` file.

## 🧪 Testing

The project includes a comprehensive test suite using Vitest.

### Running Tests

**Using Make:**
```bash
make test              # Run tests once
make test-watch        # Run tests in watch mode
make test-ui           # Run tests with UI
make test-coverage     # Run tests with coverage report
```

**Using npm:**
```bash
npm test              # Run tests once
npm run test:watch    # Run tests in watch mode
npm run test:ui       # Run tests with UI
npm run test:coverage # Run tests with coverage report
```

### Test Coverage

The test suite includes:
- **48 comprehensive tests** covering both server and frontend
- **16 server API tests** - Request validation, response format, error handling, edge cases
- **32 frontend tests** - UI interactions, card generation, navigation, security (XSS protection)

All tests run in under 500ms with 100% pass rate.

## 🐳 Docker Commands

All Docker commands are available through the Makefile:

```bash
make docker-build    # Build Docker image
make docker-up       # Start containers in detached mode
make docker-down     # Stop containers
make docker-logs     # View container logs
make docker-restart  # Restart containers
make docker-clean    # Remove containers, volumes, and images
```

Or use docker-compose directly:
```bash
docker-compose build
docker-compose up -d
docker-compose down
docker-compose logs -f
```

## 🔨 Make Commands Reference

Run `make help` to see all available commands:

```bash
make help            # Show all available commands
make install         # Install dependencies
make test            # Run tests
make test-watch      # Run tests in watch mode
make test-ui         # Run tests with UI
make test-coverage   # Run tests with coverage
make dev             # Start development server
make clean           # Clean node_modules and coverage
make docker-build    # Build Docker image
make docker-up       # Start Docker containers
make docker-down     # Stop Docker containers
make docker-logs     # View Docker logs
make docker-clean    # Remove Docker resources
make docker-restart  # Restart Docker containers
```

## 📝 Development

### Adding New Features

1. **Frontend**: Edit `public/index.html` and `public/script.js`
2. **Backend**: Modify `server.js` for API changes
3. **Styling**: Update Tailwind classes in HTML or add custom CSS
4. **Tests**: Add tests to `tests/server.test.js` or `tests/script.test.js`

### Debugging

The server includes comprehensive logging:
- All requests are logged with timestamps
- AI provider responses (OpenAI or Gemini) are logged
- Error details are logged to console
- Cost estimation for API usage

Check the terminal running `node server.js` (or `make dev`) for detailed logs.

## 🌟 Features Roadmap

- [ ] User authentication and cloud storage
- [ ] Deck management (organize cards into collections)
- [ ] Spaced repetition algorithm
- [ ] Export/Import flashcards (JSON, CSV, Anki format)
- [ ] Collaborative decks
- [ ] Mobile app (React Native)
- [ ] Multiple language support
- [ ] Voice-to-text topic input

## 📄 License

ISC

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com) for providing GPT models via their API
- [Google](https://google.com) for the Gemini API
- [Tailwind CSS](https://tailwindcss.com) for the amazing utility-first CSS framework
- [Express.js](https://expressjs.com) for the robust web framework

---

**Happy Learning! 📚✨**

If you find this project helpful, don't forget to give it a ⭐️!

# 🧠 AI Flashcard Generator

An intelligent flashcard generation application powered by OpenAI's advanced language models that creates personalized study materials from any topic. Perfect for students, educators, and lifelong learners looking to enhance their study sessions with AI-generated content.

## ✨ Features

- **🤖 AI-Powered Generation**: Leverages OpenAI's specific models to create high-quality, educational flashcards
- **🎯 Context-Aware**: Avoids repeating previously studied concepts when generating new cards
- **🎨 Beautiful Dark Mode UI**: Modern, premium interface built with Tailwind CSS
- **🔄 Interactive Flip Animation**: Smooth card flip effects for an engaging study experience
- **📊 Progress Tracking**: Keep track of how many cards you've studied
- **🔍 Search & Filter**: Quickly find specific flashcards in your collection
- **💾 Local Storage**: Your flashcards are saved locally in your browser
- **⚡ Real-time Generation**: Fast flashcard creation with loading animations
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Tech Stack

### Backend
- **Node.js** with **Express.js** - Fast, minimalist web framework
- **OpenAI API** - Advanced AI model for intelligent content generation
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
- **OpenAI API Key** - [Get one here](https://platform.openai.com/api-keys)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-flashcard-generator
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

4. **Configure your `.env` file**
   
   Open `.env` and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-5.2
   PORT=3000
   ```

## 🏃 Running the Application

1. **Start the server**
   ```bash
   node server.js
   ```

2. **Open your browser**
   
   Navigate to [http://localhost:3000](http://localhost:3000)

3. **Start generating flashcards!**
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

Generates flashcards for a given topic using OpenAI's API.

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
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ No | `gpt5.2` | OpenAI model to use |
| `PORT` | ❌ No | `3000` | Server port |

### Tailwind CSS

The project uses Tailwind CSS 4.x. The configuration can be found in `tailwind.config.js`.

## 🐛 Troubleshooting

### Port Already in Use
```
ERROR: Port 3000 is already in use!
```
**Solution**: Change the `PORT` in your `.env` file or kill the process using port 3000.

### OpenAI API Key Missing
```
OPENAI_API_KEY missing
```
**Solution**: Ensure your `.env` file exists and contains a valid `OPENAI_API_KEY`.

### Cards Not Generating
- Check your internet connection
- Verify your OpenAI API key is valid and has credits
- Check the browser console and server logs for error messages

## 📝 Development

### Adding New Features

1. **Frontend**: Edit `public/index.html` and `public/script.js`
2. **Backend**: Modify `server.js` for API changes
3. **Styling**: Update Tailwind classes in HTML or add custom CSS

### Debugging

The server includes comprehensive logging:
- All requests are logged with timestamps
- OpenAI API responses are logged
- Error details are logged to console

Check the terminal running `node server.js` for detailed logs.

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

- [OpenAI](https://openai.com) for providing the AI API
- [Tailwind CSS](https://tailwindcss.com) for the amazing utility-first CSS framework
- [Express.js](https://expressjs.com) for the robust web framework

---

**Happy Learning! 📚✨**

If you find this project helpful, don't forget to give it a ⭐️!

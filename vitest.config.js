import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    globals: true,
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true
      }
    },
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'tests/',
        '*.config.js',
        'public/styles.css',
        'src/input.css'
      ],
      include: ['server.js', 'public/script.js'],
      all: true,
      lines: 80,
      functions: 80,
      branches: 80,
      statements: 80
    },
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});

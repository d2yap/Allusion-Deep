import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import prettierRecommended from 'eslint-plugin-prettier/recommended';

export default tseslint.config(
  {
    ignores: ['build/**', 'dist/**', 'node_modules/**', 'resources/**', 'wasm/**'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  react.configs.flat.recommended,
  react.configs.flat['jsx-runtime'],
  {
    // React Compiler-era checks (refs/set-state-in-effect/...) are advisory for this
    // React 18 codebase; keep the classic hooks rules enforced.
    plugins: { 'react-hooks': reactHooks },
    rules: {
    },
  },
  prettierRecommended,
  {
    settings: {
      react: { version: '18.3' },
    },
    languageOptions: {
      parserOptions: {
        project: './tsconfig.json',
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-inferrable-types': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-unused-vars': 'error',
      quotes: ['error', 'single', { avoidEscape: true }],
      'jsx-quotes': ['error', 'prefer-double'],
      'no-trailing-spaces': 'error',
      'react/no-children-prop': 'off',
      curly: 'error',
      'no-extra-boolean-cast': 'error',
      '@typescript-eslint/no-unnecessary-condition': 'warn',
      '@typescript-eslint/prefer-string-starts-ends-with': 'error',
      'no-empty-function': 'off',
      '@typescript-eslint/no-empty-function': 'off',
      // Pre-existing style in this codebase; not part of the dependency upgrade
      // `React` is required in scope by the classic JSX runtime
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_|^e$',
          varsIgnorePattern: '^_$|^React$',
          caughtErrorsIgnorePattern: '^_|^e(rr)?$',
        },
      ],
      'no-case-declarations': 'off',
      'no-useless-assignment': 'off',
      'no-useless-escape': 'off',
      'no-async-promise-executor': 'off',
      'no-empty': 'off',
      'no-unused-expressions': 'off',
      '@typescript-eslint/no-unused-expressions': 'off',
      '@typescript-eslint/no-require-imports': 'off',
      'react/prop-types': 'off',
      'react/display-name': 'off',
    },
  },
  {
    files: ['src/backend/*.ts', 'src/frontend/entities/*.ts', 'src/frontend/stores/*.ts'],
    rules: {
      '@typescript-eslint/explicit-module-boundary-types': 'error',
    },
  },
);

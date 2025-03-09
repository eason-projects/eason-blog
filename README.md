# Eason's Technical Blog

This is a personal technical blog built using [Docusaurus](https://docusaurus.io/), a modern static website generator. The blog focuses on machine learning, software development, and other technical topics.

### Installation

```
$ pnpm install
```

### Local Development

```
$ pnpm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ pnpm build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch, using GitHub Actions. The deployment workflow is defined in `.github/workflows/deploy.yml`.

You can also manually deploy the site using:

```
$ GIT_USER=<Your GitHub username> pnpm deploy
```

This command is a convenient way to build the website and push to the `gh-pages` branch.

## Content

The blog contains articles on various technical topics including:
- Machine Learning (MLflow, Ray Tune, Optuna)
- Reinforcement Learning
- Natural Language Processing
- BLE Beacons
- Development Environment Setup
- And more

Feel free to explore the `blog` directory to see all available articles.

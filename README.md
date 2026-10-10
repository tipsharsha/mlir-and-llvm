<img src="assets/icon.svg" width="64" height="64" alt="">

# MLIR and LLVM

Training program: MLIR and LLVM for the Modular stack. It is a small local web app. Created in [T3 Code](https://t3.codes).

## Start the app

```
npm start
```

The app opens in your browser at `http://localhost:4318/` (it uses the next free port if that one is busy). No `npm install` is needed. You only need Node.js 18 or later.

The home page loads first. The tabs on the left are the weeks, and each week has chapters. Your progress is saved in `data/progress.json`, which git ignores. If you open the app without the server, it tells you what to do.

## Where the content is

```
index.html            app shell
src/app.js, app.css   app code and style
server.js             local server (static files and the progress file)
content/manifest.json the list of tabs (weeks) and chapters
content/home/         home page
content/program/      introduction, schedule, applicability and revisions
content/week-00/      Week 0: set-up, reading guide, primer, digests of R1 and R2, check
content/week-01/      Week 1: C++ refresher, LLVM IR, check
content/research/     Qualcomm context, research directions, a first project
content/reference/    session preparation, glossary, list of references
docs/archive/         the two earlier standalone HTML documents (replaced by the app)
```

## Add a week

1. Make a folder `content/week-NN/` with one HTML file for each chapter.
2. Replace the planned entry for that week in `content/manifest.json` with a `chapters` list.

A chapter file is plain HTML. Use the same classes as the existing chapters: `.task` with `data-task`, `data-title` and `data-time` for tasks that have steps, and `article[data-read]` for reading cards. The app adds the check boxes and the record sheets.

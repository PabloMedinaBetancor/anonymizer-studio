# Work with AI assistants

An assistant that writes code against Anonymizer Studio from memory gets a few things wrong, always the same few, and some of them fail without an error: a photo comes out with its plates still visible and the call succeeds. This page is for the person who reads what the assistant wrote: which files to give it, how the usual mistakes show, and how to check the code. The rules for the assistant itself are in `AGENTS.md`, at the root of the repository.

## Give the assistant the right files

- `AGENTS.md` is written for a coding assistant that has nothing else: the common tasks as whole scripts, the fields, the error codes and the mistakes. Many assistants look for it on their own inside the repository; if yours does not, name it in your request. For a project outside the repository, copy it into the project.
- `llms.txt` is an index of the documentation, one line per page.
- `llms-full.txt` is every page of the documentation as one plain file. Attach it when the assistant cannot read the repository.

## The mistakes, and how they show

| The assistant writes | What happens | What it should do |
|---|---|---|
| `anonymize_image` in a loop over a folder | It works, and loads the models again for every file | One `Anonymizer`, and `analyze` then `render` for each file |
| No look at the `Result` | Doubtful items that are not placed like a plate or a face stay visible, with no error | Read `visible_check` and `warnings`; `Options(unsure="blur")` or `set_blur` if that is the policy |
| An output name that exists, or in a folder that does not | `OutputExists`, or `FileNotFoundError`; nothing is overwritten | Make the folder first, and a new name or folder for each run |
| `out.png` for a JPEG | A JPEG named `out.png` | The extension of the original; `.mp4` for a video |
| `analyze` again "to apply" a decision | The new analysis replaces the decisions | `set_blur` or `add_area`, then `render` the same analysis |
| A relative path, then `load_analysis` from another folder | `FileNotFoundError` on the original | An absolute path to `analyze` |
| One `Anonymizer` shared between threads | Not safe | One per thread; a `CancelToken` is the one object to share |
| `analysis.preview(0)` after changing `extra_margin` | The preview ignores the change | `options=anon.options`: the default is `Options()` |
| `from anonymizer.contract import ModelLoadError` | It catches nothing the API raises | `anonymizer.ModelLoadError` |
| A claim that the report proves nothing is left | The report lists what was found, not what was missed | Look at the output; `Analysis.add_area` for what is missed |

## Check the code you were given

1. It imports only `anonymizer`, and no name outside `anonymizer.__all__`.
2. It makes one `Anonymizer` for a batch, in a `with` block, and creates the output folder.
3. In a batch, it lets `ModelLoadError` and `ValueError` stop the run and records any other `AnonymizerError` for its file.
4. It reads `visible_check` or `warnings`, or sets `unsure` on purpose.
5. It does not turn a doubtful item off on its own authority. A blur that was not needed costs a little; an item left visible by mistake publishes a face or a plate.
6. It never publishes the comparison video, which shows the original.
7. It runs: ask for the output of one run on a sample photo before you trust a long job.

A request that works names the files, the policy and the limits, and what to read first:

```text
Read AGENTS.md in this repository first. Write a script that anonymises every photo in
./incoming into ./outgoing with one Anonymizer. Blur every doubtful item (unsure="blur").
Skip files that are not photos and print their error code at the end. Do not use
anonymize_image in a loop, do not import from anonymizer.engine, and print how many
doubtful items each photo had.
```

# Anonymizer Studio

Anonymizer Studio blurs faces and licence plates in photos and videos on your own computer, and gives a person the chance to check what the detectors were not sure about before anything is published. It writes the result at the original resolution, with a privacy report of every item found and what was done with it.

It is for people who publish footage of places where strangers walk by and cars are parked, such as estate agents and street and drone videographers. It has a local web application, the Studio, and a Python API that does everything the Studio does.

![The Review screen of the Studio on a street clip, with the items found outlined on the picture and a timeline of them.](assets/first-video-decided.png)

Footage: "Calle Preciados Madrid" by OsvaldoGago, CC BY-SA 3.0, cut and re-encoded ([Licence and credits](licence-and-credits.md)).

**Read the limits first.** A face or a plate that the detectors do not find is neither blurred nor listed, and the report does not say what was missed. [Accuracy and limits](accuracy.md) says how often that happens, with the condition of every figure. The product was tried on Linux only.

## Where to go

Start with [Getting started](getting-started.md), then [Your first video](tutorials/first-video.md). [How it works](concepts/how-it-works.md) and [Privacy](concepts/privacy.md) say what it blurs and why, and [FAQ and troubleshooting](faq.md) helps when something goes wrong. Code readers, people or AI assistants, start at `AGENTS.md` in the repository root, then [Photos in Python](tutorials/photos-in-python.md) and the [reference](reference/index.md).

More in the repository: `CHANGELOG.md` and `docs/release-notes.md` (what changed), `docs/MEASURED.md` (every measurement), `docs/MISSES.md` (the missed faces) and `docs/PRIVACY-PATHS.md` (the privacy audit). This documentation is for version 2.2.0.

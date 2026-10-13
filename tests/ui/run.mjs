import { readdirSync } from "fs";
import { run } from "./harness.mjs";

const here = new URL(".", import.meta.url).pathname;
for (const file of readdirSync(here).filter((name) => name.endsWith(".test.mjs")).sort()) {
  await import(`./${file}`);
}
process.exit((await run()) === 0 ? 0 : 1);

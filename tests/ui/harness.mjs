// node here is 14, which has no built-in test runner: a minimal one with the same test() shape.
const cases = [];

export function test(name, body) {
  cases.push({ name, body });
}

export async function run() {
  let failed = 0;
  for (const { name, body } of cases) {
    try {
      await body();
      console.log(`ok   ${name}`);
    } catch (failure) {
      failed += 1;
      console.log(`FAIL ${name}\n${failure.stack}`);
    }
  }
  console.log(`\n${cases.length - failed}/${cases.length} passed`);
  return failed;
}

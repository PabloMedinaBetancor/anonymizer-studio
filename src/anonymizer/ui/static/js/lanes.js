// Greedy interval packing: each item goes in the first row of its lane that is free at its first frame.
export function packRows(items) {
  const ordered = [...items].sort((a, b) => a.first_frame - b.first_frame || a.last_frame - b.last_frame);
  const rowEnds = { face: [], plate: [], manual: [] };
  const placed = [];
  for (const item of ordered) {
    const lane = item.kind === "face" ? "face" : item.kind === "manual" ? "manual" : "plate";  // an area the person added has its own row
    const ends = rowEnds[lane];
    let row = ends.findIndex((end) => end < item.first_frame);
    if (row === -1) {
      row = ends.length;
      ends.push(item.last_frame);
    } else {
      ends[row] = item.last_frame;
    }
    placed.push({ item, lane, row });
  }
  return { placed, rows: { face: rowEnds.face.length, plate: rowEnds.plate.length, manual: rowEnds.manual.length } };
}

export function isNotSure(item) {
  return item.judgement.verdict === "check";
}

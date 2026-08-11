/** Derive a project name from the imported file names: common prefix plus
 *  the numeric range, e.g. PANO_0001.JPG..PANO_0033.JPG -> PANO_0001-0033. */
export function deriveProjectName(names: string[]): string {
  if (names.length === 0) return "panorama";
  const stems = names.map((n) => n.replace(/\.[^.]+$/, ""));
  if (stems.length === 1) return sanitizeProjectName(stems[0]!);

  const sorted = [...stems].sort();
  const a = sorted[0]!;
  const b = sorted[sorted.length - 1]!;
  let prefix = "";
  for (let i = 0; i < a.length && a[i] === b[i]; i++) prefix += a[i];

  // The prefix may end mid-number ("PANO_00"); pull those digits back.
  const m = prefix.match(/^(.*?)(\d*)$/)!;
  const base = (m[1] ?? "").replace(/[-_ .]+$/, "");
  const lo = (m[2] ?? "") + (a.slice(prefix.length).match(/^\d+/)?.[0] ?? "");
  const hi = (m[2] ?? "") + (b.slice(prefix.length).match(/^\d+/)?.[0] ?? "");
  if (base && lo && hi && lo !== hi) {
    return sanitizeProjectName(`${base}_${lo}-${hi}`);
  }
  return sanitizeProjectName(base || stems[0]!);
}

/** File-system safe: strip path separators and control characters. */
export function sanitizeProjectName(name: string): string {
  const clean = name
    // eslint-disable-next-line no-control-regex
    .replace(/[/\\:*?"<>|\x00-\x1f]/g, "")
    .trim()
    .slice(0, 80);
  return clean || "panorama";
}

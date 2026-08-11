/**
 * Session persistence in IndexedDB so a closed tab doesn't lose work.
 * `files` holds the original File blobs (written once per import);
 * `state` holds the small stuff — alignment JSON, control points, masks,
 * name — rewritten after every milestone. Best-effort: private windows
 * or quota errors silently disable it.
 */

export interface SessionShot {
  id: number;
  fileName: string;
  fullWidth: number;
  fullHeight: number;
  focalLength35mm: number | null;
  posePrior: [number, number, number] | null;
}

export interface SessionState {
  savedAt: number;
  projectName: string;
  nameEdited: boolean;
  workScale: number | null;
  shots: SessionShot[];
  alignmentJson: string | null;
  cps: unknown[] | null;
  masks: { id: number; width: number; height: number; data: Uint8Array }[];
}

const DB = "panoloom-session";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore("files");
      req.result.createObjectStore("state");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(
  db: IDBDatabase,
  store: string,
  mode: IDBTransactionMode,
  run: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const req = run(t.objectStore(store));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveFile(id: number, file: File): Promise<void> {
  try {
    // Stored as raw bytes: WebKit (and Safari private windows) cannot
    // structured-clone Blob/File into IndexedDB.
    const buf = await file.arrayBuffer();
    const db = await open();
    await tx(db, "files", "readwrite", (s) =>
      s.put({ name: file.name, type: file.type, buf }, id),
    );
    db.close();
  } catch {
    // Best-effort.
  }
}

export async function deleteFile(id: number): Promise<void> {
  try {
    const db = await open();
    await tx(db, "files", "readwrite", (s) => s.delete(id));
    db.close();
  } catch {
    // Best-effort.
  }
}

export async function saveState(state: SessionState): Promise<void> {
  try {
    const db = await open();
    await tx(db, "state", "readwrite", (s) => s.put(state, "last"));
    db.close();
  } catch {
    // Best-effort.
  }
}

export async function loadSession(): Promise<
  (SessionState & { files: Map<number, File> }) | null
> {
  try {
    const db = await open();
    const state = await tx<SessionState | undefined>(db, "state", "readonly", (s) =>
      s.get("last"),
    );
    if (!state || state.shots.length === 0) {
      db.close();
      return null;
    }
    const files = new Map<number, File>();
    for (const shot of state.shots) {
      const rec = await tx<
        { name: string; type: string; buf: ArrayBuffer } | undefined
      >(db, "files", "readonly", (s) => s.get(shot.id));
      if (!rec) {
        db.close();
        return null; // incomplete — treat as no session
      }
      files.set(shot.id, new File([rec.buf], rec.name, { type: rec.type }));
    }
    db.close();
    return { ...state, files };
  } catch {
    return null;
  }
}

export async function clearSession(): Promise<void> {
  try {
    const db = await open();
    await tx(db, "files", "readwrite", (s) => s.clear());
    await tx(db, "state", "readwrite", (s) => s.clear());
    db.close();
  } catch {
    // Best-effort.
  }
}

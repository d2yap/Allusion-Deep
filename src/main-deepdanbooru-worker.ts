import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import readline, { Interface } from 'readline';

export type TaggedResult = {
  tags: Array<{ tag: string; score: number }>;
  scores: Record<string, number>;
  truncated: Record<string, number>;
  error?: string;
};

type WorkerResponse = {
  id?: number;
  ready?: boolean;
  device?: string;
  results?: Record<string, TaggedResult>;
  failures?: Record<string, string>;
  error?: string;
  elapsed_ms?: number;
};

type Pending = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason: Error) => void;
  timer: NodeJS.Timeout;
};

/** Python 3.13 environment with onnxruntime-gpu, created by `yarn setup:gpu`. */
const DEFAULT_VENV_PYTHON = path.join(os.homedir(), 'venvs', 'allusion-gpu', 'bin', 'python');

const LOAD_TIMEOUT_MS = 180_000;
const REQUEST_TIMEOUT_MS = 120_000;

function firstExisting(paths: string[]): string {
  return paths.find((candidate) => candidate && fs.existsSync(candidate)) || '';
}

/**
 * Keeps one Python helper process alive so the tagger model is loaded once instead of per image.
 * Requests carry an id so replies can be matched without relying on ordering.
 */
export class TaggerWorker {
  private child: ChildProcessWithoutNullStreams | null = null;
  private lines: Interface | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private readyPromise: Promise<WorkerResponse> | null = null;
  private currentDevice = '';
  private scriptPath = '';
  private modelPath = '';
  private tagListPath = '';
  readonly cudaPythonPath: string;

  constructor(cudaPythonPath?: string) {
    this.cudaPythonPath = cudaPythonPath || '';
  }

  get device(): string {
    return this.currentDevice;
  }

  get isRunning(): boolean {
    return this.child !== null;
  }

  private resolvePython(): { pythonPath: string; useCuda: boolean } {
    const configured = process.env.DEEPDANBOORU_PYTHON;
    if (configured) {
      return { pythonPath: configured, useCuda: false };
    }
    const cuda = firstExisting([
      this.cudaPythonPath,
      process.env.DEEPDANBOORU_CUDA_PYTHON || '',
      DEFAULT_VENV_PYTHON,
    ]);
    return cuda ? { pythonPath: cuda, useCuda: true } : { pythonPath: 'python', useCuda: false };
  }

  /** Starts the worker if needed; resolves once the model is loaded. */
  ensureRunning(
    scriptPath: string,
    modelPath: string,
    tagListPath: string,
  ): Promise<WorkerResponse> {
    if (this.readyPromise) {
      return this.readyPromise;
    }
    this.scriptPath = scriptPath;
    this.modelPath = modelPath;
    this.tagListPath = tagListPath;

    const { pythonPath, useCuda } = this.resolvePython();
    const args = [
      scriptPath,
      '--serve',
      '--prewarm',
      '--device',
      useCuda ? 'auto' : 'cpu',
      '--model',
      modelPath,
      '--taglist',
      tagListPath,
    ];
    console.info('Starting persistent tagger worker', { pythonPath, args });

    const child = spawn(pythonPath, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    this.child = child;
    // node typings here predate the web ReadableStream-based overloads
    this.lines = (readline.createInterface as any)({ input: child.stdout }) as Interface;

    this.readyPromise = new Promise<WorkerResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.readyPromise = null;
        reject(new Error('Tagger worker did not become ready in time'));
      }, LOAD_TIMEOUT_MS);
      const lines = this.lines as Interface;
      lines.once('line', (line) => {
        clearTimeout(timer);
        try {
          const message = JSON.parse(line) as WorkerResponse;
          if (message.ready) {
            this.currentDevice = message.device || 'unknown';
            console.info('Tagger worker ready on', this.currentDevice);
            resolve(message);
          } else {
            reject(new Error(message.error || 'Tagger worker failed to start'));
          }
        } catch (err) {
          reject(new Error(`Malformed worker handshake: ${line}`));
        }
      });
    });

    this.lines.on('line', (line) => this.handleLine(line));
    child.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString().trim();
      if (text) {
        console.warn('Tagger worker stderr:', text);
      }
    });
    child.on('close', (code) =>
      this.handleClose(new Error(`Tagger worker exited with code ${code}`)),
    );
    child.on('error', (err) => this.handleClose(err));

    return this.readyPromise;
  }

  private handleLine(line: string) {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let message: WorkerResponse;
    try {
      message = JSON.parse(trimmed) as WorkerResponse;
    } catch (err) {
      console.warn('Tagger worker produced non-JSON output:', trimmed);
      return;
    }
    if (message.id === undefined) {
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timer);
    this.pending.delete(message.id);
    pending.resolve(message);
  }

  private handleClose(err: Error) {
    this.lines?.close();
    this.lines = null;
    this.child = null;
    this.readyPromise = null;
    this.currentDevice = '';
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(err);
    }
    this.pending.clear();
  }

  stop() {
    if (this.child && this.child.stdin.writable) {
      try {
        this.child.stdin.write(JSON.stringify({ op: 'shutdown' }) + '\n');
      } catch (err) {
        // worker already gone
      }
    }
    this.handleClose(new Error('Tagger worker stopped'));
  }

  /** Restarts with the same paths, e.g. after the user picks a different model. */
  restart() {
    const { scriptPath, modelPath, tagListPath } = this;
    this.stop();
    if (scriptPath && modelPath && tagListPath) {
      void this.ensureRunning(scriptPath, modelPath, tagListPath).catch((err) =>
        console.warn('Could not restart tagger worker', err),
      );
    }
  }

  async runBatch(
    imagePaths: string[],
    generalThresh: number,
  ): Promise<Record<string, TaggedResult>> {
    if (imagePaths.length === 0) {
      return {};
    }
    if (!this.child) {
      await this.ensureRunning(this.scriptPath, this.modelPath, this.tagListPath);
    }
    const child = this.child;
    if (!child) {
      throw new Error('Tagger worker is not running');
    }
    const id = this.nextId++;
    const response = await new Promise<WorkerResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('Tagger worker request timed out'));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(
        JSON.stringify({ id, op: 'predict', images: imagePaths, general_thresh: generalThresh }) +
          '\n',
      );
    });

    const out: Record<string, TaggedResult> = {};
    for (const [imagePath, result] of Object.entries(response.results || {})) {
      out[imagePath] = result;
    }
    for (const [imagePath, failure] of Object.entries(response.failures || {})) {
      out[imagePath] = { tags: [], scores: {}, truncated: {}, error: failure };
    }
    if (response.error) {
      throw new Error(response.error);
    }
    return out;
  }
}

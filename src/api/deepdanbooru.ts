import { ipcRenderer } from 'electron';
import {
  DEEPDANBOORU_PREDICT,
  DEEPDANBOORU_PREDICT_REPLY,
  DeepDanbooruPredictMessage,
  DeepDanbooruPredictReply,
} from '../ipc/messages';

export const DEFAULT_TAG_THRESHOLD = 0.5;

let nextRequestId = 1;

export async function predictDeepDanbooru(
  imagePath: string,
  projectPath: string,
  threshold = DEFAULT_TAG_THRESHOLD,
  requestId = nextRequestId++,
): Promise<DeepDanbooruPredictReply> {
  let useProject = projectPath;
  if (!useProject) {
    // ask main process for configured DeepDanbooru project location
    try {
      // deepdanbooru:get-config returns an object with optional projectPath/modelPath/tagListPath

      // use ipcRenderer.invoke to fetch config
      // NOTE: this may return {} if not configured

      const cfg: any = await (ipcRenderer as any).invoke('deepdanbooru:get-config');
      if (cfg && cfg.projectPath) {
        useProject = cfg.projectPath;
      }
    } catch (e) {
      console.warn('Could not read DeepDanbooru config from main process', e);
    }
  }
  const msg: DeepDanbooruPredictMessage = {
    imagePath,
    projectPath: useProject,
    threshold,
    requestId,
  };
  return new Promise((resolve) => {
    // Requests are matched by id; replies for other in-flight requests are ignored
    const listener = (_: unknown, reply: DeepDanbooruPredictReply) => {
      if (reply.requestId !== undefined && reply.requestId !== requestId) {
        return;
      }
      ipcRenderer.removeListener(DEEPDANBOORU_PREDICT_REPLY, listener);
      resolve(reply);
    };
    ipcRenderer.on(DEEPDANBOORU_PREDICT_REPLY, listener);
    ipcRenderer.send(DEEPDANBOORU_PREDICT, msg);
  });
}

export type { DeepDanbooruPredictMessage, DeepDanbooruPredictReply };

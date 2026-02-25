export type EmotionProbs = Record<string, number>;

export type FaceResult = {
  bbox?: number[]; // [x1, y1, x2, y2]
  score?: number;
  emotion?: {
    label?: string;
    prob?: number;
    probs?: EmotionProbs;
  };
  age_gender?: {
    age_bin?: string;
    age_bin_prob?: number;
    age_est?: number;
    gender?: string;
    gender_prob?: number;
  };
};

// ML может вернуть либо массив лиц, либо объект { faces: [...] }
export type MlInferResponse =
  | FaceResult[]
  | {
      faces?: FaceResult[];
      [k: string]: any;
    };

export type HistoryRecord = {
  id: string;
  createdAt: string;
  image: {
    filename: string;
    originalname?: string;
    mimetype?: string;
    size?: number;
    url: string; // абсолютный или относительный
  };
  result: MlInferResponse;
};

// Утилита для UI: всегда вернуть массив лиц
export function getFaces(result: MlInferResponse | null | undefined): FaceResult[] {
  if (!result) return [];
  if (Array.isArray(result)) return result;
  if (Array.isArray(result.faces)) return result.faces;
  return [];
}

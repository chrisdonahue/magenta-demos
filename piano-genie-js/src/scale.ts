import * as tf from '@tensorflow/tfjs-core';

export type MidiPitch = number;
export type PitchList = MidiPitch[];
export type Semitone = number;
export type IntervalList = Semitone[];

const NOTE_NAME_TO_ROOT: {[key:string]: MidiPitch} = {
    'C': 0,
    'G': 7,
    'D': 2,
    'A': 9,
    'E': 4,
    'B': 11,
    'F#': 6,
    'C#': 1,
    'G#': 8,
    'D#': 3,
    'A#': 8,
    'F': 5
}

const SCALE_NAME_TO_INTERVAL_LIST: {[key:string]: IntervalList} = {
    'Major': [2, 2, 1, 2, 2, 2, 1],
    'Minor': [2, 1, 2, 2, 1, 2, 2],
    'Blues': [3, 2, 1, 1, 3, 2],
    'Major Pentatonic': [2, 2, 3, 2, 3],
    'Minor Pentatonic': [3, 2, 2, 3, 2],
    //'Locrian': [1, 2, 2, 1, 2, 2, 2],
    //'Dorian': [2, 1, 2, 2, 2, 1, 2],
    //'Phrygian': [1, 2, 2, 2, 1, 2, 2],
    //'Lydian': [2, 2, 2, 1, 2, 2, 1],
    //'Mixolydian': [2, 2, 1, 2, 2, 1, 2],
    'Whole Tone': [2, 2, 2, 2, 2, 2],
}

export const NOTE_NAMES = Object.keys(NOTE_NAME_TO_ROOT);
export const SCALE_NAMES = Object.keys(SCALE_NAME_TO_INTERVAL_LIST);

export function getPitchList(rootName: string, scaleName: string) {
    let root = NOTE_NAME_TO_ROOT[rootName];
    let intervalList = SCALE_NAME_TO_INTERVAL_LIST[scaleName];

    while (root > 0) {
        root -= 12;
    }

    const result: PitchList = [];
    let note = root;
    let scalePosition = 0;
    while (note < 128) {
        if (note >= 21 && note <= 108) {
            result.push(note);
        }
        note += intervalList[scalePosition % intervalList.length];
        ++scalePosition;
    }

    return result;
}

export function scaleMaskFromPitchList(pitchList: PitchList) {
    const maskArr = new Float32Array(88);
    for (const pitch of pitchList) {
        maskArr[pitch - 21] = 1;
    }

    const mask2d = tf.tidy(() => {
        const mask = tf.tensor1d(maskArr, 'float32');
        return tf.reshape(mask, [1, 88]) as tf.Tensor2D;
    });

    return mask2d;
}

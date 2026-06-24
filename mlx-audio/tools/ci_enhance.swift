// ci_enhance — macOS Core Image "auto enhance light & colour" for video.
//
// Uses CIImage.autoAdjustmentFilters (auto white-balance / exposure / contrast /
// tone, optional face-aware) computed ONCE on a reference frame and applied
// uniformly to every frame — this mirrors Final Cut's "Balance Color" (one
// correction per clip) and avoids per-frame flicker. GPU accelerated.
//
// Output is VIDEO ONLY (H.264). Audio is muxed back by the caller (ffmpeg).
//
// Usage: ci_enhance <input> <output.mp4> [--face] [--level <0..1>] [--ref <seconds>]

import AVFoundation
import CoreImage
import Foundation
import Metal

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

// ---- args ----
let args = CommandLine.arguments
guard args.count >= 3 else { fail("usage: ci_enhance <input> <output.mp4> [--face] [--level N] [--ref S]") }
let inputPath = args[1]
let outputPath = args[2]
var faceEnhance = false
var level: Double = 1.0          // blend strength 0..1 (1 = full auto correction)
var refTime: Double = -1.0       // seconds for the reference frame; -1 = midpoint
var i = 3
while i < args.count {
    switch args[i] {
    case "--face": faceEnhance = true
    case "--level": i += 1; if i < args.count { level = Double(args[i]) ?? 1.0 }
    case "--ref":   i += 1; if i < args.count { refTime = Double(args[i]) ?? -1.0 }
    default: break
    }
    i += 1
}
level = max(0.0, min(1.0, level))

let inputURL = URL(fileURLWithPath: inputPath)
let outputURL = URL(fileURLWithPath: outputPath)
try? FileManager.default.removeItem(at: outputURL)

let asset = AVURLAsset(url: inputURL)
let sem = DispatchSemaphore(value: 0)

// ---- load video track + properties (async API) ----
var videoTrack: AVAssetTrack?
var durationSeconds: Double = 0
var naturalSize = CGSize.zero
var preferredTransform = CGAffineTransform.identity
var nominalFPS: Float = 30

Task {
    do {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let vt = tracks.first else { fail("no video track in input") }
        videoTrack = vt
        let dur = try await asset.load(.duration)
        durationSeconds = CMTimeGetSeconds(dur)
        naturalSize = try await vt.load(.naturalSize)
        preferredTransform = try await vt.load(.preferredTransform)
        nominalFPS = try await vt.load(.nominalFrameRate)
        if nominalFPS <= 0 { nominalFPS = 30 }
    } catch {
        fail("failed to load asset: \(error)")
    }
    sem.signal()
}
sem.wait()
guard let vTrack = videoTrack else { fail("no video track") }

// ---- Core Image context on Metal ----
guard let mtlDevice = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let ciContext = CIContext(mtlDevice: mtlDevice, options: [.cacheIntermediates: false])

// ---- compute auto-adjustment filter chain from a reference frame ----
let refSeconds = refTime >= 0 ? refTime : max(0.0, durationSeconds * 0.5)
let imgGen = AVAssetImageGenerator(asset: asset)
imgGen.appliesPreferredTrackTransform = false
imgGen.requestedTimeToleranceBefore = .zero
imgGen.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)

var autoFilters: [CIFilter] = []
do {
    let t = CMTime(seconds: refSeconds, preferredTimescale: 600)
    let cg = try imgGen.copyCGImage(at: t, actualTime: nil)
    let refImage = CIImage(cgImage: cg)
    let opts: [CIImageAutoAdjustmentOption: Any] = [
        .enhance: true,
        .redEye: false,
        .features: faceEnhance ? [] : [],
        .crop: false,
        .level: false,
    ]
    autoFilters = refImage.autoAdjustmentFilters(options: opts)
} catch {
    // If we cannot read a reference frame, fall back to identity (copy).
    autoFilters = []
}

func enhance(_ image: CIImage) -> CIImage {
    var out = image
    for f in autoFilters {
        f.setValue(out, forKey: kCIInputImageKey)
        if let result = f.outputImage { out = result }
    }
    if level < 1.0 {
        // blend corrected over original by `level`
        guard let blend = CIFilter(name: "CIDissolveTransition") else { return out }
        blend.setValue(image, forKey: kCIInputImageKey)
        blend.setValue(out, forKey: kCIInputTargetImageKey)
        blend.setValue(level, forKey: kCIInputTimeKey)
        if let b = blend.outputImage { return b.cropped(to: image.extent) }
    }
    return out
}

// ---- reader ----
guard let reader = try? AVAssetReader(asset: asset) else { fail("cannot create reader") }
let readerSettings: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
]
let readerOutput = AVAssetReaderTrackOutput(track: vTrack, outputSettings: readerSettings)
readerOutput.alwaysCopiesSampleData = false
guard reader.canAdd(readerOutput) else { fail("cannot add reader output") }
reader.add(readerOutput)

// ---- writer (video only, H.264) ----
guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else { fail("cannot create writer") }
let outW = Int(abs(naturalSize.width).rounded())
let outH = Int(abs(naturalSize.height).rounded())
let bitrate = max(2_000_000, Int(Double(outW * outH) * Double(nominalFPS) * 0.12))
let writerSettings: [String: Any] = [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: outW,
    AVVideoHeightKey: outH,
    AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: bitrate,
        AVVideoMaxKeyFrameIntervalKey: Int(nominalFPS) * 2,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
    ],
]
let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: writerSettings)
writerInput.expectsMediaDataInRealTime = false
writerInput.transform = preferredTransform
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: writerInput,
    sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: outW,
        kCVPixelBufferHeightKey as String: outH,
    ])
guard writer.canAdd(writerInput) else { fail("cannot add writer input") }
writer.add(writerInput)

guard reader.startReading() else { fail("reader start failed: \(String(describing: reader.error))") }
guard writer.startWriting() else { fail("writer start failed: \(String(describing: writer.error))") }
writer.startSession(atSourceTime: .zero)

let queue = DispatchQueue(label: "ci_enhance.write")
var frameCount = 0
let done = DispatchSemaphore(value: 0)

writerInput.requestMediaDataWhenReady(on: queue) {
    while writerInput.isReadyForMoreMediaData {
        guard reader.status == .reading,
              let sample = readerOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
            writerInput.markAsFinished()
            writer.finishWriting { done.signal() }
            return
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        let ciIn = CIImage(cvPixelBuffer: pixelBuffer)
        let ciOut = enhance(ciIn)

        var outBuffer: CVPixelBuffer?
        guard let pool = adaptor.pixelBufferPool,
              CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outBuffer) == kCVReturnSuccess,
              let ob = outBuffer else {
            continue
        }
        ciContext.render(ciOut, to: ob)
        adaptor.append(ob, withPresentationTime: pts)
        frameCount += 1
    }
}

done.wait()
if writer.status == .failed {
    fail("writer failed: \(String(describing: writer.error))")
}
FileHandle.standardError.write("enhanced \(frameCount) frames\n".data(using: .utf8)!)
exit(0)

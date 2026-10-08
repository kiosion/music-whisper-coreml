import CoreML
import Foundation

private struct CaptionRequest: Decodable {
    let features: Data
    let maxNewTokens: Int
    let batchSize: Int
}

private struct CaptionResponse: Encodable {
    let results: [CaptionResult]
    let encodingSeconds: Double
    let generationSeconds: Double
    let predictionSeconds: Double
    let samplingSeconds: Double
}

struct CaptionResult: Encodable {
    let tokens: [Int]
    let endedWithEOS: Bool
}

enum CaptionError: Error {
    case invalidArguments, invalidFeatures, invalidTokens, incompatibleModel
}

enum CaptionComputeUnits: String, CaseIterable {
    case cpuOnly = "cpu-only"
    case cpuAndGPU = "cpu-and-gpu"
    case cpuAndNeuralEngine = "cpu-and-neural-engine"
    case all

    var value: MLComputeUnits {
        switch self {
        case .cpuOnly: .cpuOnly
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .all: .all
        }
    }
}

private struct ModelManifest: Decodable {
    let version: Int
    let runtime: String
    let decoder_layers: Int
    let d_model: Int
    let max_source_positions: Int
    let max_target_positions: Int
    let audio_cache_length: Int
    let batch_size: Int
}

struct CaptionTokens {
    var inputs: [Int]
    var tokens: [[Int]]
    let maximum: Int

    init(count: Int, maximum: Int) {
        inputs = Array(repeating: 50258, count: count)
        tokens = Array(repeating: [], count: count)
        self.maximum = maximum
    }

    func active(_ row: Int) -> Bool {
        tokens[row].last != 50257 && tokens[row].count < maximum
    }

    mutating func advance(_ logits: MLMultiArray, position: Int, languages: [Int]) throws {
        for row in inputs.indices where active(row) {
            if position == 0 {
                inputs[row] = try CaptionWorker.argmax(logits, row: row, candidates: languages)
            } else if position == 1 {
                inputs[row] = 50363
            } else {
                inputs[row] = try CaptionWorker.argmax(logits, row: row, suppressFirst: position == 2)
                tokens[row].append(inputs[row])
            }
        }
    }
}

@main
struct CaptionWorker {
    static func main() async {
        do {
            try await run()
        } catch {
            FileHandle.standardError.write(Data("Core ML captions failed: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() async throws {
        guard CommandLine.arguments.count == 4,
              let computeUnits = CaptionComputeUnits(rawValue: CommandLine.arguments[3]) else {
            throw CaptionError.invalidArguments
        }
        let models = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let source = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        let configuration = try JSONSerialization.jsonObject(with: Data(contentsOf:
            source.appendingPathComponent("generation_config.json"))) as? [String: Any]
        guard let languages = configuration?["lang_to_id"] as? [String: Int],
              !languages.isEmpty, languages.values.allSatisfy({ (50259...50357).contains($0) }) else {
            throw CaptionError.invalidTokens
        }
        let manifest = try JSONDecoder().decode(ModelManifest.self,
            from: Data(contentsOf: models.appendingPathComponent("caption-coreml.json")))
        guard manifest.version == 4, manifest.runtime == "coreml", (1...4).contains(manifest.batch_size),
              (1...64).contains(manifest.decoder_layers), (1...4096).contains(manifest.d_model),
              manifest.max_source_positions == 1500, manifest.max_target_positions == 448,
              manifest.audio_cache_length == 1504 else {
            throw CaptionError.incompatibleModel
        }
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = computeUnits.value
        let encoder = try await MLModel.load(contentsOf: models.appendingPathComponent("AudioEncoder.mlmodelc"),
            configuration: modelConfiguration)
        let decoder = try await MLModel.load(contentsOf: models.appendingPathComponent("TextDecoder.mlmodelc"),
            configuration: modelConfiguration)
        try validateModels(encoder, decoder, manifest)
        let json = JSONEncoder()
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:
            ["ready": 3, "batchSize": manifest.batch_size, "computeUnits": computeUnits.rawValue]) + Data([10]))
        while let line = readLine() {
            let request = try JSONDecoder().decode(CaptionRequest.self, from: Data(line.utf8))
            guard (1...manifest.batch_size).contains(request.batchSize),
                  request.features.count == request.batchSize * 80 * 3000 * MemoryLayout<Float>.stride,
                  (1...440).contains(request.maxNewTokens) else { throw CaptionError.invalidFeatures }
            let features = try MLMultiArray(shape: [NSNumber(value: manifest.batch_size), 80, 1, 3000], dataType: .float32)
            features.dataPointer.initializeMemory(as: UInt8.self, repeating: 0,
                count: features.count * MemoryLayout<Float>.stride)
            request.features.withUnsafeBytes { bytes in
                features.dataPointer.copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            let started = Date()
            let encoded = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary:
                ["melspectrogram_features": features]))
            let state = decoder.makeState()
            for index in 0..<manifest.decoder_layers {
                for kind in ["key", "value"] {
                    let name = "audio_\(kind)_\(index)"
                    guard let projection = encoded.featureValue(for: name)?.multiArrayValue else {
                        throw CaptionError.incompatibleModel
                    }
                    try state.withMultiArray(for: name) { target in
                        guard projection.shape == target.shape,
                              projection.dataType == .float16, target.dataType == .float16 else {
                            throw CaptionError.incompatibleModel
                        }
                        projection.transfer(to: target)
                    }
                }
            }
            let encodedAt = Date()
            let generated = try await generate(decoder, state, request.maxNewTokens, languages.values.sorted(),
                count: request.batchSize, capacity: manifest.batch_size)
            let response = CaptionResponse(results: generated.tokens.map { CaptionResult(tokens: $0, endedWithEOS: $0.last == 50257) },
                encodingSeconds: encodedAt.timeIntervalSince(started), generationSeconds: Date().timeIntervalSince(encodedAt),
                predictionSeconds: generated.prediction, samplingSeconds: generated.sampling)
            FileHandle.standardOutput.write(try json.encode(response) + Data([10]))
        }
    }

    private static func validateModels(_ encoder: MLModel, _ decoder: MLModel, _ manifest: ModelManifest) throws {
        let inputs = decoder.modelDescription.inputDescriptionsByName
        let states = decoder.modelDescription.stateDescriptionsByName
        let projections = encoder.modelDescription.outputDescriptionsByName
        let batch = NSNumber(value: manifest.batch_size)
        guard Set(inputs.keys) == ["input_ids", "cache_length", "decoder_key_padding_mask", "active_rows"],
              inputs["input_ids"]?.multiArrayConstraint?.shape == [batch],
              inputs["cache_length"]?.multiArrayConstraint?.shape == [batch],
              inputs["active_rows"]?.multiArrayConstraint?.shape == [batch],
              inputs["decoder_key_padding_mask"]?.multiArrayConstraint?.shape == [batch, 448],
              encoder.modelDescription.inputDescriptionsByName["melspectrogram_features"]?.multiArrayConstraint?.shape == [batch, 80, 1, 3000],
              states.count == manifest.decoder_layers * 4, projections.count == manifest.decoder_layers * 2,
              decoder.modelDescription.outputDescriptionsByName["logits"]?.multiArrayConstraint?.shape == [batch, 1, 51865] else {
            throw CaptionError.incompatibleModel
        }
        for index in 0..<manifest.decoder_layers {
            for kind in ["key", "value"] {
                for cache in ["self", "audio"] {
                    let name = "\(cache)_\(kind)_\(index)"
                    let length = cache == "self" ? 448 : manifest.audio_cache_length
                    guard let constraint = states[name]?.stateConstraint,
                          constraint.bufferShape == [manifest.batch_size, manifest.d_model, 1, length],
                          constraint.dataType == .float16 else { throw CaptionError.incompatibleModel }
                    if cache == "audio" {
                        guard let output = projections[name]?.multiArrayConstraint,
                              output.shape.map(\.intValue) == constraint.bufferShape, output.dataType == .float16 else {
                            throw CaptionError.incompatibleModel
                        }
                    }
                }
            }
        }
    }

    static func generate(_ decoder: MLModel, _ state: MLState, _ maximum: Int,
                         _ languageTokens: [Int], count: Int, capacity: Int) async throws -> (tokens: [[Int]], prediction: Double, sampling: Double) {
        let batch = NSNumber(value: capacity)
        let inputIds = try MLMultiArray(shape: [batch], dataType: .int32)
        let cacheLength = try MLMultiArray(shape: [batch], dataType: .int32)
        let activeRows = try MLMultiArray(shape: [batch], dataType: .int32)
        let paddingMask = try MLMultiArray(shape: [batch, 448], dataType: .float16)
        paddingMask.dataPointer.assumingMemoryBound(to: Float16.self).initialize(repeating: -10000, count: capacity * 448)
        let inputs = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds,
            "cache_length": cacheLength, "decoder_key_padding_mask": paddingMask, "active_rows": activeRows])
        var decoding = CaptionTokens(count: count, maximum: maximum)
        var predictionSeconds = 0.0
        var samplingSeconds = 0.0
        for position in 0..<(maximum + 2) {
            for row in 0..<capacity {
                let active = row < count && decoding.active(row)
                inputIds[row] = NSNumber(value: active ? decoding.inputs[row] : 50257)
                cacheLength[row] = NSNumber(value: active ? position : 0)
                activeRows[row] = NSNumber(value: active ? 1 : 0)
                if active { paddingMask[row * 448 + position] = 0 }
            }
            let started = Date()
            let prediction = try await decoder.prediction(from: inputs, using: state)
            let predictedAt = Date()
            predictionSeconds += predictedAt.timeIntervalSince(started)
            guard let logits = prediction.featureValue(for: "logits")?.multiArrayValue else {
                throw CaptionError.invalidTokens
            }
            try decoding.advance(logits, position: position, languages: languageTokens)
            samplingSeconds += Date().timeIntervalSince(predictedAt)
            if decoding.inputs.indices.allSatisfy({ !decoding.active($0) }) { break }
        }
        return (decoding.tokens, predictionSeconds, samplingSeconds)
    }

    static func argmax(_ logits: MLMultiArray, row: Int, candidates: [Int]? = nil,
                       suppressFirst: Bool = false) throws -> Int {
        guard logits.dataType == .float16, logits.shape.count == 3,
              logits.shape[1] == 1, logits.shape[2] == 51865,
              (0..<logits.shape[0].intValue).contains(row) else { throw CaptionError.invalidTokens }
        let values = logits.dataPointer.assumingMemoryBound(to: Float16.self)
        let stride = logits.strides.last!.intValue
        let offset = row * logits.strides[0].intValue
        var bestValue = -Float.infinity
        var bestToken = -1
        for index in 0..<(candidates?.count ?? 51865) {
            let token = candidates?[index] ?? index
            if suppressFirst && (token == 220 || token == 50257) { continue }
            let value = Float(values[offset + token * stride])
            guard !value.isNaN else { throw CaptionError.invalidTokens }
            if value > bestValue {
                bestValue = value
                bestToken = token
            }
        }
        guard bestToken >= 0, bestValue.isFinite else { throw CaptionError.invalidTokens }
        return bestToken
    }
}

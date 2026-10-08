import CoreML
import XCTest
@testable import AnalysisCaptions

final class CaptionTokensTests: XCTestCase {
    private func logits(_ rows: [[Int: Float16]]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [NSNumber(value: rows.count), 1, 51865], dataType: .float16)
        result.dataPointer.assumingMemoryBound(to: Float16.self).initialize(repeating: -10, count: result.count)
        for (row, peaks) in rows.enumerated() {
            for (token, value) in peaks {
                result[row * 51865 + token] = NSNumber(value: Float(value))
            }
        }
        return result
    }

    func testIndependentLanguageEOSAndTokenLimit() throws {
        var batch = CaptionTokens(count: 2, maximum: 3)
        try batch.advance(logits([[50259: 3, 50260: 2, 5: 10], [50259: 2, 50260: 3]]),
            position: 0, languages: [50259, 50260])
        XCTAssertEqual(batch.inputs, [50259, 50260])
        try batch.advance(logits([[:], [:]]), position: 1, languages: [50259, 50260])
        XCTAssertEqual(batch.inputs, [50363, 50363])
        try batch.advance(logits([[50257: 10, 220: 9, 100: 5], [50257: 10, 101: 5]]),
            position: 2, languages: [50259, 50260])
        XCTAssertEqual(batch.tokens, [[100], [101]])
        try batch.advance(logits([[50257: 10], [102: 5]]), position: 3, languages: [50259, 50260])
        XCTAssertFalse(batch.active(0))
        XCTAssertTrue(batch.active(1))
        try batch.advance(logits([[999: 10], [103: 5]]), position: 4, languages: [50259, 50260])
        XCTAssertEqual(batch.tokens, [[100, 50257], [101, 102, 103]])
        XCTAssertFalse(batch.active(1))
    }

    func testSingleRowResetAndTieOrder() throws {
        for _ in 0..<2 {
            var batch = CaptionTokens(count: 1, maximum: 1)
            try batch.advance(logits([[50259: 1]]), position: 0, languages: [50259])
            try batch.advance(logits([[:]]), position: 1, languages: [50259])
            try batch.advance(logits([[301: 5, 300: 5]]), position: 2, languages: [50259])
            XCTAssertEqual(batch.tokens, [[300]])
            XCTAssertFalse(batch.active(0))
        }
    }

    func testArgmaxHonorsRowAndElementStrides() throws {
        let count = 2 * 51872 * 2
        let storage = UnsafeMutablePointer<Float16>.allocate(capacity: count)
        storage.initialize(repeating: -10, count: count)
        let value = try MLMultiArray(dataPointer: UnsafeMutableRawPointer(storage), shape: [2, 1, 51865],
            dataType: .float16, strides: [NSNumber(value: 51872 * 2), NSNumber(value: 51872 * 2), 2],
            deallocator: { $0.deallocate() })
        storage[12 * 2] = 5
        storage[51872 * 2 + 42 * 2] = 7
        XCTAssertEqual(try CaptionWorker.argmax(value, row: 0), 12)
        XCTAssertEqual(try CaptionWorker.argmax(value, row: 1), 42)
    }

    func testComputePoliciesAreExplicit() {
        XCTAssertEqual(CaptionComputeUnits(rawValue: "cpu-only")?.value, .cpuOnly)
        XCTAssertEqual(CaptionComputeUnits(rawValue: "cpu-and-gpu")?.value, .cpuAndGPU)
        XCTAssertEqual(CaptionComputeUnits(rawValue: "cpu-and-neural-engine")?.value, .cpuAndNeuralEngine)
        XCTAssertEqual(CaptionComputeUnits(rawValue: "all")?.value, .all)
        XCTAssertNil(CaptionComputeUnits(rawValue: "gpu-only"))
    }
}

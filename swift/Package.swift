// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AnalysisCaptions",
    platforms: [.macOS(.v15)],
    products: [.executable(name: "analysis-captions-coreml", targets: ["AnalysisCaptions"])],
    targets: [
        .executableTarget(name: "AnalysisCaptions"),
        .testTarget(name: "AnalysisCaptionsTests", dependencies: ["AnalysisCaptions"]),
    ]
)

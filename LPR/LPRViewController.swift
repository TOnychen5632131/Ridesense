

import UIKit
import AVFoundation
import Vision

class LPRViewController: UIViewController, AVCapturePhotoCaptureDelegate {
    
    // MARK: - Public Properties
    
    public var bufferSize: CGSize = .zero
    public var targetPlateNumber: String? {
        didSet {
            // 可以在这里添加属性观察器，在设置目标车牌时进行一些初始化操作
            if let number = targetPlateNumber {
                speakText("Starting to search for license plate: \(number)")
            }
        }
    }
    
    // MARK: - Private Properties
    
    private var lprView: LPRView!
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput",
                                                     qos: .userInitiated,
                                                     attributes: [],
                                                     autoreleaseFrequency: .workItem)
    private let photoOutput = AVCapturePhotoOutput()
    private var requests = [VNRequest]()
    private let readPlateNumberQueue = OperationQueue()
    private let licensePlateController = LicensePlateController()
    private let synthesizer = AVSpeechSynthesizer()
    private var hasFoundTarget = false // 用于追踪是否已经找到目标车牌
    private let textRecognitionQueue = DispatchQueue(label: "TextRecognitionQueue")
    private var textRecognitionRequest: VNRecognizeTextRequest!
    private var confidenceThreshold: Float = 0.7 // 置信度阈值
    private var plateValidationAttempts = 0 // 验证尝试次数
    private var lastRecognizedPlates: [String] = [] // 存储最近识别的结果
    private let maxValidationAttempts = 3 // 最大验证尝试次数
    
    
    // MARK: - View Lifecycle
    
    override func loadView() {
        // 创建并设置 LPRView
        lprView = LPRView()
        lprView.videoPlayerView.videoGravity = .resizeAspectFill
        self.view = lprView
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setUp()
        
        // 设置整个页面的无障碍标签
        view.accessibilityLabel = "License Plate Scanning Screen"
        
        // 添加返回钮
        let backButton = UIButton(type: .system)
        backButton.translatesAutoresizingMaskIntoConstraints = false
        backButton.setImage(UIImage(systemName: "xmark"), for: .normal)
        backButton.tintColor = .white
        backButton.accessibilityLabel = "Stop Scanning"
        backButton.accessibilityHint = "Double tap to stop scanning and return to previous screen"
        backButton.addTarget(self, action: #selector(backButtonTapped), for: .touchUpInside)
        
        view.addSubview(backButton)
        
        NSLayoutConstraint.activate([
            backButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            backButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            backButton.widthAnchor.constraint(equalToConstant: 44),
            backButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    @objc private func backButtonTapped() {
        captureSession.stopRunning()
        dismiss(animated: true)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(true)
        captureSession.startRunning()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        captureSession.stopRunning()
    }
    
    // MARK: - Private Methods
    
    private func setUp() {
        lprView.videoPlayerView.videoGravity = .resizeAspectFill
        setUpAVCapture()
        try? setUpVision()
    }
    
    private func setUpAVCapture() {
        var deviceInput: AVCaptureDeviceInput!
        
        // Select a video device, make an input
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        captureSession.beginConfiguration()
        
        captureSession.sessionPreset = .vga640x480 // Model image size is smaller.
        
        // Add a video input
        guard captureSession.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(deviceInput)
        
        // Add video output
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            // Add a video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            captureSession.commitConfiguration()
            return
        }
        
        let captureConnection = videoDataOutput.connection(with: .video)
        // Always process the frames
        captureConnection?.isEnabled = true
        
        // Get buffer size to allow for determining recognized license plate positions
        // relative to the video ouput buffer size
        do {
            try videoDevice!.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice!.unlockForConfiguration()
        } catch {
            print(error)
        }
        
        // Add photo output
        if captureSession.canAddOutput(photoOutput) {
            photoOutput.isHighResolutionCaptureEnabled = true
            captureSession.addOutput(photoOutput)
        }
    
        captureSession.commitConfiguration()
    
        lprView.bufferSize = bufferSize
        lprView.session = captureSession
    }
    
    private func setUpVision() throws {
        let visionModel = try VNCoreMLModel(for: LicensePlateDetector().model)
        
        let objectRecognition = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            self?.processResults(results)
        }
        
        textRecognitionRequest = VNRecognizeTextRequest { [weak self] request, error in
            guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
            self?.processRecognizedText(observations)
        }
        textRecognitionRequest.recognitionLevel = .accurate
        textRecognitionRequest.usesLanguageCorrection = false
        textRecognitionRequest.minimumTextHeight = 0.1
        textRecognitionRequest.recognitionLanguages = ["en-US"]
        textRecognitionRequest.customWords = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        self.requests = [objectRecognition]
    }
    
    private func processResults(_ results: [VNRecognizedObjectObservation]) {
        let rects = results.map {
            VNImageRectForNormalizedRect($0.boundingBox,
                                         Int(bufferSize.width),
                                         Int(bufferSize.height))
        }
        
        licensePlateController.updateLicensePlates(withRects: rects)
        
        // 在这里添加存储逻辑
        for plate in licensePlateController.licensePlates {
            if let number = plate.number {
                // 使用 UserDefaults 存储 (简单方案)
                var savedPlates = UserDefaults.standard.array(forKey: "SavedPlates") as? [String] ?? []
                if !savedPlates.contains(number) {
                    savedPlates.append(number)
                    UserDefaults.standard.set(savedPlates, forKey: "SavedPlates")
                }
            }
        }
        
        // 处理识别到的车牌
        processPlates(licensePlateController.licensePlates)
        
        // perform drawing on main thread
        DispatchQueue.main.async {
            self.lprView.licensePlates = self.licensePlateController.licensePlates
        }
        
        getPlateNumber()
    }
    
    private func processRecognizedText(_ observations: [VNRecognizedTextObservation]) {
        var recognizedPlates: [String] = []
        
        for observation in observations {
            // 获取前3个候选结果而不是只要第一个
            let candidates = observation.topCandidates(3)
            for candidate in candidates where candidate.confidence >= confidenceThreshold {
                let text = candidate.string.uppercased()
                // 清理文本，移除空格和特殊字符
                let cleanedText = text.components(separatedBy: CharacterSet.alphanumerics.inverted).joined()
                
                if isValidLicensePlate(cleanedText) {
                    recognizedPlates.append(cleanedText)
                    break // 找到有效车牌就处理下一个观察结果
                }
            }
        }
        
        // 如果有识别到的车牌
        if !recognizedPlates.isEmpty {
            lastRecognizedPlates.append(contentsOf: recognizedPlates)
            
            // 当收集到足够的样本时进行验证
            if lastRecognizedPlates.count >= maxValidationAttempts {
                if let validatedPlate = getMostFrequentPlate() {
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        if let firstPlate = self.licensePlateController.licensePlatesWithoutNumbers.first {
                            self.licensePlateController.addNumber(validatedPlate, to: firstPlate)
                            print("识别到车牌号: \(validatedPlate)") // 添加调试输出
                        }
                    }
                }
                lastRecognizedPlates.removeAll()
            }
        }
    }
    
    private func isValidLicensePlate(_ text: String) -> Bool {
        // 基本长度检查
        guard text.count >= 5 && text.count <= 8 else { return false }
        
        // 确保包含数字和字母
        let letters = text.filter { $0.isLetter }
        let numbers = text.filter { $0.isNumber }
        
        guard letters.count >= 2 && numbers.count >= 1 else { return false }
        
        // 检查是否符合常见车牌格式
        let platePattern = "^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}$"
        return text.range(of: platePattern, options: .regularExpression) != nil
    }
    
    private func getMostFrequentPlate() -> String? {
        let groupedPlates = Dictionary(grouping: lastRecognizedPlates, by: { $0 })
        let sortedPlates = groupedPlates.sorted { $0.value.count > $1.value.count }
        
        // 只有当最频繁出现的结果出现次数超过总样本的一半时才返回
        if let mostFrequent = sortedPlates.first,
           mostFrequent.value.count > maxValidationAttempts / 2 {
            return mostFrequent.key
        }
        return nil
    }
    
    private func getPlateNumber() {
        guard let firstPlate = licensePlateController.licensePlatesWithoutNumbers.first,
              readPlateNumberQueue.operationCount == 0 else {
            return
        }
        
        let rect = firstPlate.lastRectInBuffer
        let regionOfInterest = CGRect(x: rect.minX / bufferSize.width,
                                     y: rect.minY / bufferSize.height,
                                     width: rect.width / bufferSize.width,
                                     height: rect.height / bufferSize.height)
        
        // 创建照片设置
        let photoSettings = AVCapturePhotoSettings()
        photoSettings.isHighResolutionPhotoEnabled = true
        
        // 捕获高质量照片
        photoOutput.capturePhoto(with: photoSettings, delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput,
                    didFinishProcessingPhoto photo: AVCapturePhoto,
                    error: Error?) {
        guard error == nil,
              let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData),
              let cgImage = image.cgImage else { return }
        
        // 使用预处理增强图像
        let processedImage = preprocessImage(cgImage) ?? cgImage
        
        guard let firstPlate = licensePlateController.licensePlatesWithoutNumbers.first else { return }
        let rect = firstPlate.lastRectInBuffer
        let regionOfInterest = CGRect(x: rect.minX / bufferSize.width,
                                     y: rect.minY / bufferSize.height,
                                     width: rect.width / bufferSize.width,
                                     height: rect.height / bufferSize.height)
        
        // 创建多个识别请求，使用不同的参数
        let requests = createTextRecognitionRequests(firstPlate: firstPlate)
        
        // 执行所有请求
        let handler = VNImageRequestHandler(cgImage: processedImage, options: [:])
        try? handler.perform(requests)
    }
    
    // 添加新方法：创建多个识别请求
    private func createTextRecognitionRequests(firstPlate: LicensePlate) -> [VNRecognizeTextRequest] {
        var requests: [VNRecognizeTextRequest] = []
        
        // 请求1：标准精确识别
        let request1 = VNRecognizeTextRequest { [weak self] request, error in
            self?.handleRecognizedText(request, firstPlate: firstPlate)
        }
        request1.recognitionLevel = .accurate
        request1.minimumTextHeight = 0.1
        request1.recognitionLanguages = ["en-US"]
        requests.append(request1)
        
        // 请求2：快速识别
        let request2 = VNRecognizeTextRequest { [weak self] request, error in
            self?.handleRecognizedText(request, firstPlate: firstPlate)
        }
        request2.recognitionLevel = .fast
        request2.minimumTextHeight = 0.08
        request2.recognitionLanguages = ["en-US"]
        requests.append(request2)
        
        return requests
    }
    
    // 添加新方法：处理识别结果
    private func handleRecognizedText(_ request: VNRequest, firstPlate: LicensePlate) {
        guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
        
        var possiblePlates: [(text: String, confidence: Float)] = []
        
        for observation in observations {
            // 获取更多候选结果
            let candidates = observation.topCandidates(5)
            for candidate in candidates {
                let text = candidate.string.uppercased()
                    .components(separatedBy: CharacterSet.alphanumerics.inverted)
                    .joined()
                
                // 使用更宽松的验证规则
                if isSimpleLicensePlateFormat(text) {
                    possiblePlates.append((text, candidate.confidence))
                }
            }
        }
        
        // 按置信度排序并更新
        if let bestPlate = possiblePlates.sorted(by: { $0.confidence > $1.confidence }).first {
            DispatchQueue.main.async { [weak self] in
                self?.licensePlateController.addNumber(bestPlate.text, to: firstPlate)
                print("识别到车牌号: \(bestPlate.text) 置信度: \(bestPlate.confidence)")
            }
        }
    }
    
    // 添加新方法：更宽松的车牌格式验证
    private func isSimpleLicensePlateFormat(_ text: String) -> Bool {
        // 基本长度检查
        guard text.count >= 4 && text.count <= 8 else { return false }
        
        // 确保同时包含字母和数字
        let hasLetters = text.contains { $0.isLetter }
        let hasNumbers = text.contains { $0.isNumber }
        
        return hasLetters && hasNumbers
    }
    
    private func processPlates(_ plates: [LicensePlate]) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.lprView.licensePlates = plates
            
            if let targetPlate = self.targetPlateNumber {
                for plate in plates {
                    if let number = plate.number, 
                       number.contains(targetPlate), 
                       !self.hasFoundTarget {
                        self.hasFoundTarget = true
                        self.playTargetFoundAlert()
                        
                        UIAccessibility.post(notification: .announcement, argument: "Target vehicle found: \(number)")
                        self.speakText("Target vehicle found: \(number)")
                        
                        let foundPlateLabel = UILabel()
                        foundPlateLabel.text = number
                        foundPlateLabel.isAccessibilityElement = true
                        foundPlateLabel.accessibilityLabel = "Found license plate: \(number)"
                        foundPlateLabel.accessibilityTraits = .staticText
                        
                        break
                    }
                }
            }
        }
    }
    
    private func playTargetFoundAlert() {
        // 播放系统提示音
        AudioServicesPlaySystemSound(1016) // 系统提示音
        
        // 触发震动
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
        
        // 额外的震动反馈
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    }
    
    private func speakText(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        synthesizer.speak(utterance)
    }
    
    // 优化图像预处理方法
    private func preprocessImage(_ image: CGImage) -> CGImage? {
        let ciImage = CIImage(cgImage: image)
        let context = CIContext()
        
        // 1. 自适应直方图均衡化
        guard let histogramFilter = CIFilter(name: "CIHistogramEqualization") else { return nil }
        histogramFilter.setValue(ciImage, forKey: "inputImage")
        
        // 2. 增强对比度和亮度
        guard let colorControls = CIFilter(name: "CIColorControls") else { return nil }
        colorControls.setValue(histogramFilter.outputImage, forKey: "inputImage")
        colorControls.setValue(1.5, forKey: "inputContrast") // 增加对比度
        colorControls.setValue(0.0, forKey: "inputSaturation") // 移除颜色
        colorControls.setValue(0.2, forKey: "inputBrightness") // 提高亮度
        
        // 3. 锐化
        guard let sharpenFilter = CIFilter(name: "CISharpenLuminance") else { return nil }
        sharpenFilter.setValue(colorControls.outputImage, forKey: "inputImage")
        sharpenFilter.setValue(1.5, forKey: "inputSharpness")
        
        // 4. 降噪（轻微）
        guard let noiseReductionFilter = CIFilter(name: "CINoiseReduction") else { return nil }
        noiseReductionFilter.setValue(sharpenFilter.outputImage, forKey: "inputImage")
        noiseReductionFilter.setValue(0.01, forKey: "inputNoiseLevel")
        noiseReductionFilter.setValue(0.8, forKey: "inputSharpness")
        
        guard let outputImage = noiseReductionFilter.outputImage,
              let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }
        
        return cgImage
    }
}

// MARK: - Video Data Output Delegate

extension LPRViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                        orientation: .currentRearCameraOrientation,
                                                        options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput,
                       didDrop didDropSampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // print("frame dropped")
    }
}

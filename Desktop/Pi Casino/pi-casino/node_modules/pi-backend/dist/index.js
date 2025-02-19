"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var StellarSdk = __importStar(require("stellar-sdk"));
var utils_1 = require("./utils");
var PiNetwork = /** @class */ (function () {
    function PiNetwork(apiKey, walletPrivateSeed, options) {
        if (options === void 0) { options = null; }
        var _this = this;
        this.createPayment = function (paymentData) { return __awaiter(_this, void 0, void 0, function () {
            var axiosClient, body, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        this.validatePaymentData(paymentData);
                        axiosClient = (0, utils_1.getAxiosClient)(this.API_KEY, this.axiosOptions);
                        body = { payment: paymentData };
                        return [4 /*yield*/, axiosClient.post("/v2/payments", body)];
                    case 1:
                        response = _a.sent();
                        this.currentPayment = response.data;
                        return [2 /*return*/, response.data.identifier];
                }
            });
        }); };
        this.submitPayment = function (paymentId) { return __awaiter(_this, void 0, void 0, function () {
            var _a, txid_1, errorObject, _b, amount, paymentIdentifier, fromAddress, toAddress, piHorizon, transactionData, transaction, txid;
            var _c, _d;
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0:
                        _e.trys.push([0, , 5, 6]);
                        if (!(!this.currentPayment || this.currentPayment.identifier != paymentId)) return [3 /*break*/, 2];
                        _a = this;
                        return [4 /*yield*/, this.getPayment(paymentId)];
                    case 1:
                        _a.currentPayment = _e.sent();
                        txid_1 = (_d = (_c = this.currentPayment) === null || _c === void 0 ? void 0 : _c.transaction) === null || _d === void 0 ? void 0 : _d.txid;
                        if (txid_1) {
                            errorObject = { message: "This payment already has a linked txid", paymentId: paymentId, txid: txid_1 };
                            throw new Error(JSON.stringify(errorObject));
                        }
                        _e.label = 2;
                    case 2:
                        _b = this.currentPayment, amount = _b.amount, paymentIdentifier = _b.identifier, fromAddress = _b.from_address, toAddress = _b.to_address;
                        piHorizon = this.getHorizonClient(this.currentPayment.network);
                        transactionData = {
                            amount: amount,
                            paymentIdentifier: paymentIdentifier,
                            fromAddress: fromAddress,
                            toAddress: toAddress,
                        };
                        return [4 /*yield*/, this.buildA2UTransaction(piHorizon, transactionData)];
                    case 3:
                        transaction = _e.sent();
                        return [4 /*yield*/, this.submitTransaction(piHorizon, transaction)];
                    case 4:
                        txid = _e.sent();
                        return [2 /*return*/, txid];
                    case 5:
                        this.currentPayment = null;
                        return [7 /*endfinally*/];
                    case 6: return [2 /*return*/];
                }
            });
        }); };
        this.completePayment = function (paymentId, txid) { return __awaiter(_this, void 0, void 0, function () {
            var axiosClient, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, , 2, 3]);
                        axiosClient = (0, utils_1.getAxiosClient)(this.API_KEY, this.axiosOptions);
                        return [4 /*yield*/, axiosClient.post("/v2/payments/".concat(paymentId, "/complete"), { txid: txid })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.data];
                    case 2:
                        this.currentPayment = null;
                        return [7 /*endfinally*/];
                    case 3: return [2 /*return*/];
                }
            });
        }); };
        this.getPayment = function (paymentId) { return __awaiter(_this, void 0, void 0, function () {
            var axiosClient, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        axiosClient = (0, utils_1.getAxiosClient)(this.API_KEY, this.axiosOptions);
                        return [4 /*yield*/, axiosClient.get("/v2/payments/".concat(paymentId))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.data];
                }
            });
        }); };
        this.cancelPayment = function (paymentId) { return __awaiter(_this, void 0, void 0, function () {
            var axiosClient, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, , 2, 3]);
                        axiosClient = (0, utils_1.getAxiosClient)(this.API_KEY, this.axiosOptions);
                        return [4 /*yield*/, axiosClient.post("/v2/payments/".concat(paymentId, "/cancel"))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.data];
                    case 2:
                        this.currentPayment = null;
                        return [7 /*endfinally*/];
                    case 3: return [2 /*return*/];
                }
            });
        }); };
        this.getIncompleteServerPayments = function () { return __awaiter(_this, void 0, void 0, function () {
            var axiosClient, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        axiosClient = (0, utils_1.getAxiosClient)(this.API_KEY, this.axiosOptions);
                        return [4 /*yield*/, axiosClient.get("/v2/payments/incomplete_server_payments")];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.data];
                }
            });
        }); };
        this.validateSeedFormat = function (seed) {
            if (!seed.startsWith("S"))
                throw new Error("Wallet private seed must starts with 'S'");
            if (seed.length !== 56)
                throw new Error("Wallet private seed must be 56-character long");
        };
        this.validatePaymentData = function (paymentData) {
            if (!paymentData.amount)
                throw new Error("Missing amount");
            if (!paymentData.memo)
                throw new Error("Missing memo");
            if (!paymentData.metadata)
                throw new Error("Missing metadata");
            if (!paymentData.uid)
                throw new Error("Missing uid");
        };
        this.getHorizonClient = function (network) {
            _this.NETWORK_PASSPHRASE = network;
            var serverUrl = network === "Pi Network" ? "https://api.mainnet.minepi.com" : "https://api.testnet.minepi.com";
            return new StellarSdk.Server(serverUrl);
        };
        this.buildA2UTransaction = function (piHorizon, transactionData) { return __awaiter(_this, void 0, void 0, function () {
            var myAccount, baseFee, paymentOperation, transaction, _a, _b, _c;
            var _d;
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0:
                        if (transactionData.fromAddress !== this.myKeypair.publicKey()) {
                            throw new Error("You should use a private seed of your app wallet!");
                        }
                        return [4 /*yield*/, piHorizon.loadAccount(this.myKeypair.publicKey())];
                    case 1:
                        myAccount = _e.sent();
                        return [4 /*yield*/, piHorizon.fetchBaseFee()];
                    case 2:
                        baseFee = _e.sent();
                        paymentOperation = StellarSdk.Operation.payment({
                            destination: transactionData.toAddress,
                            asset: StellarSdk.Asset.native(),
                            amount: transactionData.amount.toString(),
                        });
                        _b = (_a = StellarSdk.TransactionBuilder).bind;
                        _c = [void 0, myAccount];
                        _d = {
                            fee: baseFee.toString(),
                            networkPassphrase: this.NETWORK_PASSPHRASE
                        };
                        return [4 /*yield*/, piHorizon.fetchTimebounds(180)];
                    case 3:
                        transaction = new (_b.apply(_a, _c.concat([(_d.timebounds = _e.sent(),
                                _d)])))()
                            .addOperation(paymentOperation)
                            .addMemo(StellarSdk.Memo.text(transactionData.paymentIdentifier))
                            .build();
                        transaction.sign(this.myKeypair);
                        return [2 /*return*/, transaction];
                }
            });
        }); };
        this.submitTransaction = function (piHorizon, transaction) { return __awaiter(_this, void 0, void 0, function () {
            var txResponse;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, piHorizon.submitTransaction(transaction)];
                    case 1:
                        txResponse = _a.sent();
                        // @ts-ignore
                        return [2 /*return*/, txResponse.id];
                }
            });
        }); };
        this.validateSeedFormat(walletPrivateSeed);
        this.API_KEY = apiKey;
        this.myKeypair = StellarSdk.Keypair.fromSecret(walletPrivateSeed);
        this.axiosOptions = options;
    }
    return PiNetwork;
}());
exports.default = PiNetwork;

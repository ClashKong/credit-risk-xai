import { AxiosClientOptions, PaymentArgs, PaymentDTO } from "./types";
export default class PiNetwork {
    private API_KEY;
    private myKeypair;
    private NETWORK_PASSPHRASE;
    private currentPayment;
    private axiosOptions;
    constructor(apiKey: string, walletPrivateSeed: string, options?: AxiosClientOptions | null);
    createPayment: (paymentData: PaymentArgs) => Promise<string>;
    submitPayment: (paymentId: string) => Promise<string>;
    completePayment: (paymentId: string, txid: string) => Promise<PaymentDTO>;
    getPayment: (paymentId: string) => Promise<PaymentDTO>;
    cancelPayment: (paymentId: string) => Promise<PaymentDTO>;
    getIncompleteServerPayments: () => Promise<Array<PaymentDTO>>;
    private validateSeedFormat;
    private validatePaymentData;
    private getHorizonClient;
    private buildA2UTransaction;
    private submitTransaction;
}

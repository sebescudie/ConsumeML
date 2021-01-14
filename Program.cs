using System;
using System.Dynamic;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace ConsumeML
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelPath = @"C:\Users\seb\Documents\dev\vvvv\quickies\ml_net_taxi_price\assets\Model.zip";

            // Setup stuff
            MLContext context = new MLContext();
            DataViewSchema pipeline;
            ITransformer model = context.Model.Load(modelPath, out pipeline);

            Console.WriteLine("Loaded model");

            // Create the prediction engine
            var predictionEngine = context.Model.CreatePredictionEngine<TaxiTrip, TaxiFarePrediction>(model);

            // The TaxiTrip we wanna estimate the price
            TaxiTrip inputTrip = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0
            };

            var result = predictionEngine.Predict(inputTrip);

            Console.WriteLine("Estimated cost : " + result.Score);

            TaxiTrip secondInputTrip = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 10f,
                PaymentType = "CRD",
                FareAmount = 0
            };

            TaxiFarePrediction secondResult = (TaxiFarePrediction)predictionEngine.Predict(secondInputTrip);

            Console.WriteLine("Second Estimated cost : " + secondResult.Score);

            Console.ReadKey();
        }
    }
}

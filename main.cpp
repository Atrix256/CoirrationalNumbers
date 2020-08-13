#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DETERMINISTIC() true
#define SHOW_PROGRESS() false

static const double c_pi =          3.14159265359;
static const double c_goldenRatio = 1.61803398875;
static const double c_e =           2.71828182845;
static const double c_sqrt2 =       sqrt(2.0);

static const double c_goldenRatioConjugate = 0.61803398875;

// when making a continued fraction, if the fractional remainder part is less than this, consider it zero.
static const double c_zeroThreshold = 0.000001; // 1 / 1 million

struct LabelAndNumber
{
    const char* label;
    double number;
};

std::mt19937 GetRNG()
{
#if DETERMINISTIC()
    static int seed = 1336+seed;
    seed++;
    std::mt19937 rng(seed);
#else
    std::random_device rd;
    std::mt19937 rng(rd());
#endif
    return rng;
}

typedef std::vector<int> ContinuedFraction;

ContinuedFraction ToContinuedFraction(double f, int maxContinuedFractionTerms = 15)
{
    ContinuedFraction continuedFraction;

    while (maxContinuedFractionTerms == 0 || int(continuedFraction.size()) < maxContinuedFractionTerms)
    {
        // break the number into the integer and fractional part.
        int integerPart = int(f);
        double fractionalPart = f - floor(f);

        // the integer part is the next number in the continued fraction
        continuedFraction.push_back(integerPart);

        // if no fractional part, we are done
        if (fractionalPart < c_zeroThreshold)
            break;

        // f = 1/fractionalPart and continue
        f = 1.0 / fractionalPart;
    }

    // return the "extended" form of continued fractions, that is... always make sure the CF ends in 1.
    if (/*continuedFraction.size() < maxContinuedFractionTerms &&*/ continuedFraction.back() != 1)
    {
        continuedFraction.back()--;
        continuedFraction.push_back(1);
    }

    return continuedFraction;
}

double FromContinuedFraction(const ContinuedFraction& continuedFraction, int count = 0)
{
    double ret = 0.0;
    if (count == 0)
        count = (int)continuedFraction.size();
    int index = std::min(count, (int)continuedFraction.size()) - 1;
    for (; index >= 0; --index)
    {
        if (ret != 0.0)
            ret = 1.0 / ret;
        ret += continuedFraction[index];
    }
    return ret;
}

void ToFraction(const ContinuedFraction& continuedFraction, int count, size_t& numerator, size_t& denominator)
{
    numerator = 0;
    denominator = 1;

    if (count == 0)
        count = (int)continuedFraction.size();
    int index = std::min(count, (int)continuedFraction.size()) - 1;

    for (; index >= 0; --index)
    {
        if (numerator != 0)
            std::swap(numerator, denominator);
        numerator += size_t(continuedFraction[index]) * denominator;
    }
}

// Yeah these functions are a little redundant...
// FromContinuedFraction(), ToFraction
// ToFloat() uses the left to right method and is the most recent
double ToDouble(const ContinuedFraction& continuedFraction)
{
    if (continuedFraction.size() == 0)
        return 0.0;

    int n_1 = 1;
    int d_1 = 0;

    int n_0 = continuedFraction[0];
    int d_0 = 1;

    for (size_t i = 1; i < continuedFraction.size(); ++i)
    {
        int n = continuedFraction[i] * n_0 + n_1;
        int d = continuedFraction[i] * d_0 + d_1;

        n_1 = n_0;
        d_1 = d_0;

        n_0 = n;
        d_0 = d;
    }

    return double(n_0) / double(d_0);
}

void PrintContinuedFraction(double f, const char* label = nullptr, int maxContinuedFractionTerms = 15)
{
    ContinuedFraction cf = ToContinuedFraction(f, maxContinuedFractionTerms);
    if (label)
        printf("%s = %f = [%i", label, f,  cf[0]);
    else
        printf("%f = [%i", f, cf[0]);
    for (size_t index = 1; index < cf.size(); ++index)
        printf(", %i", cf[index]);
    printf("] (rt = %f)\n", ToDouble(cf));
}

void Test_ContinuedFractionError(const char* fileName, const std::vector<LabelAndNumber>& labelsAndNumbers)
{
    FILE* file = nullptr;
    fopen_s(&file, fileName, "w+t");

    for (const LabelAndNumber& labelAndNumber : labelsAndNumbers)
    {
        fprintf(file, "\"%s\"", labelAndNumber.label);

        ContinuedFraction continuedFraction = ToContinuedFraction(labelAndNumber.number);

        for (int digits = 1; digits < continuedFraction.size(); ++digits)
        {
            double value = FromContinuedFraction(continuedFraction, digits);
            double relativeError = value / labelAndNumber.number - 1.0;
            fprintf(file, ",\"%f\"", abs(relativeError));
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// -------------------------------------------------------------------------------

struct RGB
{
    unsigned char R, G, B;
};

float SmoothStep(float value, float min, float max)
{
    float x = (value - min) / (max - min);
    x = std::min(x, 1.0f);
    x = std::max(x, 0.0f);

    return 3.0f * x * x - 2.0f * x * x * x;
}

template <typename T>
T Lerp(T A, T B, float t)
{
    return T(float(A) * (1.0f - t) + float(B) * t);
}

void DrawLine(std::vector<RGB>& image, int width, int height, int x1, int y1, int x2, int y2, RGB color)
{
    // pad the AABB of pixels we scan, to account for anti aliasing
    int startX = std::max(std::min(x1, x2) - 4, 0);
    int startY = std::max(std::min(y1, y2) - 4, 0);
    int endX = std::min(std::max(x1, x2) + 4, width - 1);
    int endY = std::min(std::max(y1, y2) + 4, height - 1);

    // if (x1,y1) is A and (x2,y2) is B, get a normalized vector from A to B called AB
    float ABX = float(x2 - x1);
    float ABY = float(y2 - y1);
    float ABLen = std::sqrtf(ABX*ABX + ABY * ABY);
    ABX /= ABLen;
    ABY /= ABLen;

    // scan the AABB of our line segment, drawing pixels for the line, as is appropriate
    for (int iy = startY; iy <= endY; ++iy)
    {
        RGB* pixel = &image[(iy * width + startX)];
        for (int ix = startX; ix <= endX; ++ix)
        {
            // project this current pixel onto the line segment to get the closest point on the line segment to the point
            float ACX = float(ix - x1);
            float ACY = float(iy - y1);
            float lineSegmentT = ACX * ABX + ACY * ABY;
            lineSegmentT = std::min(lineSegmentT, ABLen);
            lineSegmentT = std::max(lineSegmentT, 0.0f);
            float closestX = float(x1) + lineSegmentT * ABX;
            float closestY = float(y1) + lineSegmentT * ABY;

            // calculate the distance from this pixel to the closest point on the line segment
            float distanceX = float(ix) - closestX;
            float distanceY = float(iy) - closestY;
            float distance = std::sqrtf(distanceX*distanceX + distanceY * distanceY);

            // use the distance to figure out how transparent the pixel should be, and apply the color to the pixel
            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel->R = Lerp(pixel->R, color.R, alpha);
                pixel->G = Lerp(pixel->G, color.G, alpha);
                pixel->B = Lerp(pixel->B, color.B, alpha);
            }

            pixel++;
        }
    }
}

void DrawCircleFilled(std::vector<RGB>& image, int width, int height, int cx, int cy, int radius, RGB color)
{
    int startX = std::max(cx - radius - 4, 0);
    int startY = std::max(cy - radius - 4, 0);
    int endX = std::min(cx + radius + 4, width - 1);
    int endY = std::min(cy + radius + 4, height - 1);

    for (int iy = startY; iy <= endY; ++iy)
    {
        float dy = float(cy - iy);
        RGB* pixel = &image[(iy * width + startX)];
        for (int ix = startX; ix <= endX; ++ix)
        {
            float dx = float(cx - ix);

            float distance = std::max(std::sqrtf(dx * dx + dy * dy) - float(radius), 0.0f);

            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel->R = Lerp(pixel->R, color.R, alpha);
                pixel->G = Lerp(pixel->G, color.G, alpha);
                pixel->B = Lerp(pixel->B, color.B, alpha);
            }

            pixel ++;
        }
    }
}

void DrawCircle(std::vector<RGB>& image, int width, int height, int cx, int cy, int radius, RGB color)
{
    int startX = std::max(cx - radius - 4, 0);
    int startY = std::max(cy - radius - 4, 0);
    int endX = std::min(cx + radius + 4, width - 1);
    int endY = std::min(cy + radius + 4, height - 1);

    for (int iy = startY; iy <= endY; ++iy)
    {
        float dy = float(cy - iy);
        RGB* pixel = &image[(iy * width + startX)];
        for (int ix = startX; ix <= endX; ++ix)
        {
            float dx = float(cx - ix);

            float distance = abs(std::sqrtf(dx * dx + dy * dy) - float(radius));

            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel->R = Lerp(pixel->R, color.R, alpha);
                pixel->G = Lerp(pixel->G, color.G, alpha);
                pixel->B = Lerp(pixel->B, color.B, alpha);
            }

            pixel++;
        }
    }
}

template <typename T>
T Fract(T x)
{
    return x - floor(x);
}

void NumberlineAndCircleTest(const char* baseFileName, float irrational)
{
    static const int c_numFrames = 16;

    static const int c_circleImageSize = 256;
    static const int c_circleRadius = 120;

    static const int c_numberlineImageWidth = c_circleImageSize;
    static const int c_numberlineImageHeight = c_numberlineImageWidth/4;
    static const int c_numberlineStartX = c_numberlineImageWidth / 10;
    static const int c_numberlineSizeX = c_numberlineImageWidth * 8 / 10;
    static const int c_numberlineEndX = c_numberlineStartX + c_numberlineSizeX;
    static const int c_numberlineLineStartY = (c_numberlineImageHeight / 2) - 10;
    static const int c_numberlineLineEndY = (c_numberlineImageHeight / 2) + 10;

    char fileName[256];
    for (int frame = 0; frame < c_numFrames; ++frame)
    {
        std::vector<RGB> circleImageLeft(c_circleImageSize*c_circleImageSize, RGB{ 255,255,255 });
        std::vector<RGB> circleImageRight(c_circleImageSize*c_circleImageSize, RGB{ 255,255,255 });
        std::vector<RGB> numberlineImageLeft(c_numberlineImageWidth*c_numberlineImageHeight, RGB{ 255, 255, 255 });
        std::vector<RGB> numberlineImageRight(c_numberlineImageWidth*c_numberlineImageHeight, RGB{ 255, 255, 255 });

        DrawCircle(circleImageLeft, c_circleImageSize, c_circleImageSize, 128, 128, c_circleRadius, RGB{ 0,0,0 });
        DrawCircle(circleImageRight, c_circleImageSize, c_circleImageSize, 128, 128, c_circleRadius, RGB{ 0,0,0 });

        DrawLine(numberlineImageLeft, c_numberlineImageWidth, c_numberlineImageHeight, c_numberlineStartX, c_numberlineImageHeight / 2, c_numberlineEndX, c_numberlineImageHeight / 2, RGB{0, 0, 0});
        DrawLine(numberlineImageRight, c_numberlineImageWidth, c_numberlineImageHeight, c_numberlineStartX, c_numberlineImageHeight / 2, c_numberlineEndX, c_numberlineImageHeight / 2, RGB{ 0, 0, 0 });

        float value = 0.0f;
        for (int sample = 0; sample <= frame; ++sample)
        {
            float angle = value * (float)c_pi * 2.0f;

            int targetX = int(cos(angle) * float(c_circleRadius)) + 128;
            int targetY = int(sin(angle) * float(c_circleRadius)) + 128;

            unsigned char percentColor = (unsigned char)(255.0f - 255.0f * float(sample) / float(c_numFrames-1));

            RGB sampleColor = (sample == frame) ? RGB{ 255, 0, 0 } : RGB{ 192, percentColor, 0 };

            DrawLine(circleImageLeft, c_circleImageSize, c_circleImageSize, 128, 128, targetX, targetY, sampleColor);

            if (sample >= c_numFrames / 2)
                DrawLine(circleImageRight, c_circleImageSize, c_circleImageSize, 128, 128, targetX, targetY, sampleColor);

            targetX = int(value * float(c_numberlineSizeX)) + c_numberlineStartX;
            DrawLine(numberlineImageLeft, c_numberlineImageWidth, c_numberlineImageHeight, targetX, c_numberlineLineStartY, targetX, c_numberlineLineEndY, sampleColor);

            if (sample >= c_numFrames / 2)
                DrawLine(numberlineImageRight, c_numberlineImageWidth, c_numberlineImageHeight, targetX, c_numberlineLineStartY, targetX, c_numberlineLineEndY, sampleColor);

            value = Fract(value + irrational);
        }

        int outImageW = c_circleImageSize * 2;
        int outImageH = c_circleImageSize + c_numberlineImageHeight;
        std::vector<RGB> outputImage(outImageW*outImageH);

        RGB* dest = outputImage.data();
        const RGB* srcLeft = circleImageLeft.data();
        const RGB* srcRight = circleImageRight.data();
        for (int i = 0; i < c_circleImageSize; ++i)
        {
            memcpy(dest, srcLeft, c_circleImageSize * 3);
            dest += c_circleImageSize;
            srcLeft += c_circleImageSize;
            memcpy(dest, srcRight, c_circleImageSize * 3);
            dest += c_circleImageSize;
            srcRight += c_circleImageSize;
        }

        srcLeft = numberlineImageLeft.data();
        srcRight = numberlineImageRight.data();
        for (int i = 0; i < c_numberlineImageHeight; ++i)
        {
            memcpy(dest, srcLeft, c_circleImageSize * 3);
            dest += c_circleImageSize;
            srcLeft += c_circleImageSize;
            memcpy(dest, srcRight, c_circleImageSize * 3);
            dest += c_circleImageSize;
            srcRight += c_circleImageSize;
        }

        sprintf_s(fileName, "out/%s_%i.png", baseFileName, frame);
        stbi_write_png(fileName, outImageW, outImageH, 3, outputImage.data(), outImageW * 3);
    }
}

void Test(float irrational)
{
    printf("%f\n", irrational);
    ContinuedFraction CF = ToContinuedFraction(irrational);
    for (size_t i = 1; i < CF.size(); ++i)
    {
        size_t n, d;
        ToFraction(CF, (int)i, n, d);
        double value = FromContinuedFraction(CF, (int)i);
        double percentError = abs(value - irrational) / irrational;
        printf("[%i] %f aka %zu/%zu (%f)\n", CF[i - 1], value, n, d, percentError);
    }
    printf("\n");
}

void Test2D(const char* baseFileName, int numSamples, float ix, float iy)
{
    const int c_imageSize = 1024;
    const int c_paddingSize = c_imageSize / 10;
    const int c_paddingStart = c_paddingSize;
    const int c_paddingEnd = c_imageSize - c_paddingStart;
    const int c_canvasSize = c_paddingEnd - c_paddingStart;
    const int c_sampleSize = c_imageSize / 128;

    std::vector<RGB> image(c_imageSize * c_imageSize, RGB{ 255, 255, 255 });

    FILE* file = nullptr;
    char fileName[256];
    sprintf_s(fileName, "out/%s.csv", baseFileName);
    fopen_s(&file, fileName, "w+t");

    float valuex = 0.0f;
    float valuey = 0.0f;
    for (int sample = 0; sample < numSamples; ++sample)
    {
        valuex = Fract(valuex + ix);
        valuey = Fract(valuey + iy);
        fprintf(file, "\"%f\",\"%f\",\n", valuex, valuey);

        int px = int(valuex * float(c_canvasSize)) + c_paddingStart;
        int py = int(valuey * float(c_canvasSize)) + c_paddingStart;

        unsigned char percentColor = (unsigned char)(255.0f * float(sample) / float(numSamples - 1));

        RGB color{ percentColor, 255, 0 };

        DrawCircle(image, c_imageSize, c_imageSize, px, py, c_sampleSize, color);

        //DrawLine(image, c_imageSize, c_imageSize, 0, 0, pointx, pointy, color);
    }
    fclose(file);

    DrawLine(image, c_imageSize, c_imageSize, c_paddingStart, c_paddingStart, c_paddingEnd, c_paddingStart, RGB{ 0, 0, 0 });
    DrawLine(image, c_imageSize, c_imageSize, c_paddingStart, c_paddingEnd, c_paddingEnd, c_paddingEnd, RGB{ 0, 0, 0 });
    DrawLine(image, c_imageSize, c_imageSize, c_paddingStart, c_paddingStart, c_paddingStart, c_paddingEnd, RGB{ 0, 0, 0 });
    DrawLine(image, c_imageSize, c_imageSize, c_paddingEnd, c_paddingStart, c_paddingEnd, c_paddingEnd, RGB{ 0, 0, 0 });

    sprintf_s(fileName, "out/%s.png", baseFileName);
    stbi_write_png(fileName, c_imageSize, c_imageSize, 3, image.data(), c_imageSize * 3);
}

void Test2D(const char* baseFileName, int numSamples, float ix)
{
    Test2D(baseFileName, numSamples, ix, ix * float(Fract(c_goldenRatio)));
}

// find <numDynamic> numbers that are maximally co-irrational to each other and to the <fixed> values
std::vector<double> FindCoIrrationals(const std::vector<double>& fixed, size_t numDynamic)
{
    std::mt19937 rng = GetRNG();

    // TODO: are all these fields needed?
    struct RandomWeight
    {
        size_t dynamicIndexSrc;
        size_t dynamicIndexTarget;
        double cfValue;
        ContinuedFraction continuedFraction;
        size_t slotIndex;
        float weight;
    };

    struct RatioItem
    {
        double ratio;
        ContinuedFraction continuedFraction;
    };

    struct DynamicItemEntry
    {
        double value;
        std::vector<RatioItem> fixedRatios;
        std::vector<RatioItem> dynamicRatios;
    };

    // make some dynamic values to start with
    std::vector<DynamicItemEntry> dynamicValues(numDynamic);
    for (size_t i = 0; i < numDynamic; ++i)
    {
        std::uniform_real_distribution<double> unidist(0.0, 1.0);
        //dynamicValues[i].value = double(i + 0.5) / double(numDynamic);
        dynamicValues[i].value = unidist(rng);
        dynamicValues[i].fixedRatios.resize(fixed.size());
        dynamicValues[i].dynamicRatios.resize(numDynamic);
    }

    // TODO: could start one dynamic value at the golden ratio perhaps.

    // TOOD: how to decrease values? decriment? what if it's like 100k or something? could maybe do a log based thing.
    // TODO: make this be a setting/parameter, or find some terminating condition...
    // TODO: maybe keep track of best score seen?
    // TODO: could try gradient descent. could also try simulated annealing.
    // TODO: report maximum irrationality? how do yo udo that? maybe csv to graph this somehow...
    static const size_t c_stepCount = 100000;
    static const float c_slotIndexWeightingMultiplier = 10.0f;

#if SHOW_PROGRESS()
    printf("fixed:\n");
    for (size_t targetIndex = 0; targetIndex < fixed.size(); ++targetIndex)
        PrintContinuedFraction(fixed[targetIndex]);
    printf("dynamic:\n");
    for (size_t targetIndex = 0; targetIndex < dynamicValues.size(); ++targetIndex)
        PrintContinuedFraction(dynamicValues[targetIndex].value);
    printf("\n");
#endif

    // iteratively improve the co-irrationality of the numbers
    std::vector<RandomWeight> randomWeights;
    for (size_t stepIndex = 0; stepIndex < c_stepCount; ++stepIndex)
    {
        // clear it out, but the internal size should keep a high water mark.
        randomWeights.resize(0);

        // make continued fractions for all ratios between the dynamic numbers and all the other numbers
        // also gather up scores at the same time
        float totalWeight = 0.0f;
        for (size_t dynamicIndex = 0; dynamicIndex < dynamicValues.size(); ++dynamicIndex)
        {
            DynamicItemEntry& dynamic = dynamicValues[dynamicIndex];

            for (size_t targetIndex = 0; targetIndex < fixed.size(); ++targetIndex)
            {
                RatioItem& targetRatio = dynamic.fixedRatios[targetIndex];

                targetRatio.ratio = dynamic.value / fixed[targetIndex];
                targetRatio.continuedFraction = ToContinuedFraction(targetRatio.ratio);

                // add a 0 so it has a chance to change to a 1
                targetRatio.continuedFraction.push_back(0);

#if SHOW_PROGRESS()
                printf("%f to %f (dynamic %zu to static %zu)\n", dynamic.value, fixed[targetIndex], dynamicIndex, targetIndex);
                PrintContinuedFraction(targetRatio.ratio);
#endif

                for (size_t slotIndex = 1; slotIndex < targetRatio.continuedFraction.size(); ++slotIndex)
                {
                    int diff = abs(targetRatio.continuedFraction[slotIndex] - 1);
                    if (diff > 0)
                    {
                        float weight = float(diff) / (float(slotIndex) * c_slotIndexWeightingMultiplier);
                        randomWeights.push_back({ dynamicIndex, (size_t)-1, targetRatio.ratio, targetRatio.continuedFraction, slotIndex, weight });
                        totalWeight += weight;
                    }
                }
            }

            for (size_t targetIndex = dynamicIndex + 1; targetIndex < dynamicValues.size(); ++targetIndex)
            {
                RatioItem& targetRatio = dynamic.dynamicRatios[targetIndex];

                targetRatio.ratio = dynamic.value / dynamicValues[targetIndex].value;
                targetRatio.continuedFraction = ToContinuedFraction(targetRatio.ratio);

                // add a 0 so it has a chance to change to a 1
                targetRatio.continuedFraction.push_back(0);

#if SHOW_PROGRESS()
                printf("%f to %f (dynamic %zu to dynamic %zu)\n", dynamic.value, dynamicValues[targetIndex].value, dynamicIndex, targetIndex);
                PrintContinuedFraction(targetRatio.ratio);
#endif

                for (size_t slotIndex = 1; slotIndex < targetRatio.continuedFraction.size(); ++slotIndex)
                {
                    int diff = abs(targetRatio.continuedFraction[slotIndex] - 1);
                    if (diff > 0)
                    {
                        float weight = float(diff) / (float(slotIndex) * c_slotIndexWeightingMultiplier);
                        randomWeights.push_back({ dynamicIndex, targetIndex, targetRatio.ratio, targetRatio.continuedFraction, slotIndex, weight });
                        totalWeight += weight;
                    }
                }
            }
        }

        // TODO: in a 2 dynamic count setup, the last number is basically never getting modified. is that a bug?

        // roll a random number to see which slot of which continued fraction should be improved
        std::uniform_real_distribution<float> dist(0.0f, totalWeight);
        float targetWeight = dist(rng);
        for (size_t randomWeightIndex = 0; randomWeightIndex < randomWeights.size(); ++randomWeightIndex)
        {
            targetWeight -= randomWeights[randomWeightIndex].weight;
            if (targetWeight > 0.0f && randomWeightIndex < randomWeights.size() - 1)
                continue;

            RandomWeight& randomWeight = randomWeights[randomWeightIndex];
            ContinuedFraction& continuedFraction = randomWeight.continuedFraction;

            double oldRatio = ToDouble(continuedFraction);

            if (continuedFraction[randomWeight.slotIndex] == 0)
                continuedFraction[randomWeight.slotIndex] = 1;
            else if (continuedFraction[randomWeight.slotIndex] > 10)
                continuedFraction[randomWeight.slotIndex] /= 2;
            else
                continuedFraction[randomWeight.slotIndex]--;

            // remove the last digit if it's a zero. we put it there just so it could be incremented if needed
            if (continuedFraction.back() == 0)
                continuedFraction.pop_back();

            double newRatio = ToDouble(continuedFraction);

            // TODO: temp
            ContinuedFraction test = ToContinuedFraction(newRatio);

            // TODO: newRatio is negative? how does that work... maybe super huge num/den?

            // adjust the value
            std::uniform_int_distribution<int> distBool(0, 1);
            if (randomWeight.dynamicIndexTarget == (size_t)-1 || distBool(rng) == 0)
                dynamicValues[randomWeight.dynamicIndexSrc].value = Fract(dynamicValues[randomWeight.dynamicIndexSrc].value * newRatio / randomWeight.cfValue);
            else
                dynamicValues[randomWeight.dynamicIndexTarget].value = Fract(dynamicValues[randomWeight.dynamicIndexTarget].value * randomWeight.cfValue / newRatio);


#if SHOW_PROGRESS()
            printf("\nfixed:\n");
            for (size_t targetIndex = 0; targetIndex < fixed.size(); ++targetIndex)
                PrintContinuedFraction(fixed[targetIndex]);
            printf("dynamic:\n");
            for (size_t targetIndex = 0; targetIndex < dynamicValues.size(); ++targetIndex)
                PrintContinuedFraction(dynamicValues[targetIndex].value);
            printf("\n");
#endif
            int ijklz = 0;
            break;
        }

        int ijkl = 0;
    }

    // TODO: either my code is wrong, or there are more than 2 ways to write rational numbers with continued fractions.
    // TODO: it might be that it's wrong how you are making sure it ends with 1?

    // make and return the vector of dynamic values
    {
        std::vector<double> ret(numDynamic);
        for (size_t i = 0; i < numDynamic; ++i)
            ret[i] = dynamicValues[i].value;
        return ret;
    }
}

void ReportCoIrrationals(const std::vector<double>& fixed, const std::vector<double>& dynamic)
{
    printf("fixed:\n");
    for (double d : fixed)
        printf("  %f\n", d);

    printf("found:\n");
    for (double d : dynamic)
        printf("  %f\n", d);

    for (size_t i = 0; i < fixed.size() + dynamic.size() - 1; ++i)
    {
        double A = (i < fixed.size()) ? fixed[i] : dynamic[i - fixed.size()];
        for (size_t j = i + 1; j < fixed.size() + dynamic.size(); ++j)
        {
            double B = (j < fixed.size()) ? fixed[j] : dynamic[j - fixed.size()];

            double ratio = (A > B) ? B / A: A / B;
            printf("[%zu to %zu] %f to %f = %f\n", i, j, A, B, ratio);
            PrintContinuedFraction(ratio);
        }
    }


    printf("\n");
}

int main(int argc, char** argv)
{

    // test number round trips
    if (false)
    {
        double value = 3.14;
        ContinuedFraction cf = ToContinuedFraction(value);
        double valueRT = ToDouble(cf);
        int ijkl = 0;
    }

    //PrintContinuedFraction(c_pi, "pi");
    //PrintContinuedFraction(c_pi / c_goldenRatio, "pi/gr");

    //Test(c_pi);
    //Test(c_pi / c_goldenRatio);

    // find a value that is coirrational with 1.0
    {
        std::vector<double> fixedValues = { 1.0 };
        std::vector<double> coirrationals = FindCoIrrationals(fixedValues, 1);

        ReportCoIrrationals(fixedValues, coirrationals);

        // TODO: numberline?
        //Test("test_1", 64, (float)coirrationals[0]);
    }

    // find 2 values that are coirrational with each other and also 1.0
    if (true)
    {
        std::vector<double> fixedValues = { 1.0 };
        std::vector<double> coirrationals = FindCoIrrationals(fixedValues, 2);

        ReportCoIrrationals(fixedValues, coirrationals);

        Test2D("test_1_2", 64, (float)coirrationals[0], (float)coirrationals[1]);
    }

    system("pause");
    return 0;

    static const int c_numPoints = 64;

    Test2D("gr_sqrt2", 64, (float)Fract(c_goldenRatio), (float)Fract(c_sqrt2));

    Test2D("pi_auto", 64, (float)Fract(c_pi));

    Test2D("gr_auto", 64, (float)Fract(c_goldenRatio));

    Test2D("1_64_auto", 64, 1.0f / 64.0f);
    Test2D("1_32_auto", 64, 1.0f / 32.0f);
    Test2D("1_16_auto", 64, 1.0f / 16.0f);
    Test2D("1_8_auto", 64, 1.0f / 8.0f);
    Test2D("1_4_auto", 64, 1.0f / 4.0f);
    Test2D("1_2_auto", 64, 1.0f / 2.0f);
    Test2D("1_1_auto", 64, 1.0f / 1.0f);

    {
        float g = 1.32471795724474602596f;
        float a1 = 1.0f / g;
        float a2 = 1.0f / (g*g);
        Test2D("R2", 64, a1, a2);
    }


    system("pause");
    return 0;
}

/*

! try an alternate algorithm: randomly pick a pair of numbers A or B. divide and make a CF. randomly cut one term in half. apply changes to A or B randomly.


! make sure it can find the golden ratio from 1 static, 1 dynamic, before moving on.

* to visualize progress, could show these on a numberline (how many samples?) in different colors and make an animated gif of them becoming co-irrational

TODO:
* try doing just 1 dynamic 1 static. dynamic should become golden ratio if it's working right.
* when deciding to improve a continued fraction, you should flip a coin to see which of the 2 numbers involved should be adjusted. (need to store both indices)
* the weighting for a digit should be based on the magnitude of change, which is based on the fractions that came before it.

TODO:

! fract(goldenRatio) and 1.0 - fract(goldenRatio) are just as irrational as eachother. what's that mean for trying to find the most irrational?
 * also kinda explains why GR and 1.0-GR pair up.

* show coirrational points as 2d?
 * pi/goldenRatio
 * show like goldenratio & golden ratio, even thought you have a random offset to the axes?

? i don't really know how to show this 2 co-irrational number setup.
 * other than like... divide both numbers by the first number and plot the second number on a numberline... but that doesn't tell us anything new

* need to figure out how to make 3 or more co-irrational numbers

* understand this from r4 unit: https://twitter.com/R4_Unit/status/1284588140473155585?s=20

* clean up this code - remove what isn't needed

Blog:
* you can tell how co-irrational 2 numbers are by dividing one by the other (doesn't matter which is which) and seeing how rational the result is. golden ratio as a result == maximally irrational.
 * all these numbers (even GR) are mod 1 of course. fractional part
* you can make a number maximally co-irrational to another by multiplying it by the golden ratio.
* um... golden ratio and it's most coirrational add up to 1.0.  that seems weird but i think maybe it's just correct. nothing ever said it needed to be coirrational with a rational number, so it's just diagonal.

*/
import express from "express";
import path from "path";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Increase payload limit for base64 images uploads
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Shared lazy-initialized Gemini client
let aiInstance: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not configured in the environment. Please add it via the Settings secrets panel.");
    }
    aiInstance = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiInstance;
}

// 1. Analyze Document Endpoint (Handles Text, OCR of Images, and Multilingual settings)
app.post("/api/analyze", async (req, res) => {
  try {
    const { text, title, imageBase64, imageMimeType, language } = req.body;
    const ai = getGeminiClient();

    let targetLanguageText = language || "English";

    // Prepare contents
    let contents: any[] = [];
    let systemInstruction = `You are an elite legal education assistant named LegalMind. 
Your goal is to parse the contract/document and translate its contents/terms into plain-English (or the requested target translation language), identify risks (Green/Yellow/Red clauses), recommendations, and summarize logically.
Do NOT give formal professional legal representation or advice; instead, maintain a supportive, educational, objective, and neutral tone. Include a reminder about the legal disclaimer in your interpretations.

IMPORTANT - Response Language guidance:
If the target language requested by the user is not English (e.g., Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, or Urdu), translate the 'simplifiedText' property, the 'summary' fields, and 'recommendations' values into that specific target language. Keep the 'originalText' string in its source original format. If the target language is English, keep everything in English.`;

    if (imageBase64 && imageMimeType) {
      // Multimodal scenario (OCR + Analysis in one pass)
      contents.push({
        inlineData: {
          mimeType: imageMimeType,
          data: imageBase64,
        },
      });
      contents.push({
        text: `The user has uploaded a scanned document or screenshot image of a contract. 
1. Perform high-accuracy OCR to extract the complete raw text of the document and assign it to the 'rawText' response property.
2. Formulate a comprehensive Document Analysis based on the extracted text.
3. Keep the target language for explanations as: ${targetLanguageText}.
Provide full, rich, detailed lists of clauses, summaries, and recommendations.`,
      });
    } else {
      // Pasted Text scenario
      const textToAnalyze = text || "";
      if (!textToAnalyze.trim()) {
        return res.status(400).json({ error: "No document text or image found to analyze." });
      }
      contents.push({
        text: `Here is the legal document text to analyze:
---
${textToAnalyze}
---
Perform a thorough analysis. 
1. Populate 'rawText' with the exactly provided raw input text.
2. Analyze the document, extract important clauses, identify risk scores, write plain simplified explanations, and make detailed checklists.
3. Keep the translation target language as: ${targetLanguageText} for simplified explanations, summaries, and recommendation items.`,
      });
    }

    // Call Gemini with schema configuration
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: contents,
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          required: ["documentType", "language", "riskScore", "riskLabel", "summary", "clauses", "recommendations", "rawText"],
          properties: {
            documentType: {
              type: Type.STRING,
              description: "The specific type of legal document analyzed (e.g. 'Non-Disclosure Agreement', 'Residential Lease Agreement', 'Freelance Contract', 'Employment Agreement', 'Terms of Service').",
            },
            language: {
              type: Type.STRING,
              description: "The original native language detected in the document.",
            },
            riskScore: {
              type: Type.INTEGER,
              description: "An overall calculated risk score from 0 (completely standard/safe) to 100 (highly toxic, one-sided, or risky).",
            },
            riskLabel: {
              type: Type.STRING,
              description: "A summary risk label matching the riskScore. E.g., 'Very Safe' (score 0-25), 'Moderate Risk' (score 26-65), or 'High Risk' (score 66-100).",
            },
            rawText: {
              type: Type.STRING,
              description: "The exact extracted text of the document from the image scanner (OCR) or standard input text.",
            },
            summary: {
              type: Type.OBJECT,
              required: ["short", "medium", "detailed", "bulletPoints"],
              properties: {
                short: { type: Type.STRING, description: "A one or two sentence high-level elevator summary." },
                medium: { type: Type.STRING, description: "A 1-paragraph explanation of the scope and core purpose." },
                detailed: { type: Type.STRING, description: "A detailed 2-3 paragraph breakdown of rights, terms, and obligations." },
                bulletPoints: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "List of key takeaways of the document.",
                },
              },
            },
            clauses: {
              type: Type.ARRAY,
              description: "List of important clauses found in the document.",
              items: {
                type: Type.OBJECT,
                required: ["category", "originalText", "simplifiedText", "rating", "explanation"],
                properties: {
                  category: { type: Type.STRING, description: "Clause category (e.g., Termination, Liability, Indemnification, Intellectual Property, Dispute Resolution, Fees)." },
                  originalText: { type: Type.STRING, description: "The literal original legal phrasing or exact sentences from the text." },
                  simplifiedText: { type: Type.STRING, description: "The converted simple, clear, layperson language explaining the clause." },
                  rating: {
                    type: Type.STRING,
                    description: "GREEN (completely normal and fair), YELLOW (needs review, watch out for auto-renewals or moderate obligations), or RED (dangerously unfair, hidden penalties, extreme one-sided liabilities, or loss of baseline user rights).",
                  },
                  explanation: { type: Type.STRING, description: "A detailed description explaining exactly WHY this rating was assigned, what risks are involved, and how it impacts the user." },
                },
              },
            },
            recommendations: {
              type: Type.OBJECT,
              required: ["thingsToAsk", "thingsToNegotiate", "importantQuestions", "missingInformation", "possibleRisks", "suggestedNextSteps"],
              properties: {
                thingsToAsk: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Questions to ask the counterparty or drafting party for clarity." },
                thingsToNegotiate: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Specific terms to negotiate or modify to protect yourself." },
                importantQuestions: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Fundamental questions the user should answer before signing." },
                missingInformation: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Crucial details normally present in such contracts that are completely missing here." },
                possibleRisks: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Potential negative consequences or legal scenarios to watch out for." },
                suggestedNextSteps: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Concrete, actionable recommendations of what to do next (e.g. consult professional lawyer, review clause #3, request digital copy)." },
              },
            },
          },
        },
      },
    });

    const resultText = response.text;
    if (!resultText) {
      throw new Error("No analysis response returned from the model.");
    }

    const parsedResult = JSON.parse(resultText.trim());
    
    // Add additional fields used by the client
    const finalResult = {
      ...parsedResult,
      id: "doc_" + Date.now(),
      title: title || parsedResult.documentType || "Unnamed Document",
      uploadedAt: new Date().toISOString(),
    };

    res.json(finalResult);
  } catch (error: any) {
    console.error("Analysis Error:", error);
    res.status(500).json({ error: error?.message || "An error occurred during legal document analysis." });
  }
});

// 2. Document Comparison Endpoint (Compares Contract A and Contract B)
app.post("/api/compare", async (req, res) => {
  try {
    const { doc1Text, doc1Title, doc2Text, doc2Title } = req.body;
    if (!doc1Text || !doc2Text) {
      return res.status(400).json({ error: "Both documents are required for comparison." });
    }

    const ai = getGeminiClient();
    const systemInstruction = `You are an elite legal document examiner. 
Your task is to comprehensively compare two agreements (Doc 1: "${doc1Title || 'Document 1'}" vs Doc 2: "${doc2Title || 'Document 2'}").
Analyze differences in clauses, addition/removal of clauses, payment terms, risk shifts, and liabilities. 
Provide a detailed structural comparison returning valid JSON.`;

    const promptText = `Please compare the following two documents:
Document 1 Title: ${doc1Title || "Document 1"}
Document 1 Text:
---
${doc1Text}
---

Document 2 Title: ${doc2Title || "Document 2"}
Document 2 Text:
---
${doc2Text}
---

Analyze the structural differences, payment rules, risks, added/removed items, and translate them into simple English. Write the differences clearly.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: promptText,
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          required: ["doc1Title", "doc2Title", "summary", "overallRiskChange", "differences", "addedClauses", "removedClauses", "paymentDifferences", "riskChanges"],
          properties: {
            doc1Title: { type: Type.STRING },
            doc2Title: { type: Type.STRING },
            summary: { type: Type.STRING, description: "A high level overview comparing both contracts (which is more favorable, what are the primary changes)." },
            overallRiskChange: { type: Type.STRING, description: "E.g. 'Safer' (risk went down), 'Moderate Increase' (some clauses added that create caution), 'Critical - Much Higher Risk' (extreme shift in liability to user)." },
            differences: {
              type: Type.ARRAY,
              description: "Detailed item-by-item structural differences.",
              items: {
                type: Type.OBJECT,
                required: ["title", "description", "status"],
                properties: {
                  title: { type: Type.STRING, description: "E.g. Termination Notice Period, Intellectual Property Ownership." },
                  description: { type: Type.STRING, description: "Explanation of how they compare in simple words." },
                  status: { type: Type.STRING, description: "Must be: 'added', 'removed', 'modified', or 'unchanged'." },
                  doc1Text: { type: Type.STRING, description: "Relevant snippet from Doc 1, if any." },
                  doc2Text: { type: Type.STRING, description: "Relevant snippet from Doc 2, if any." },
                },
              },
            },
            addedClauses: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Clauses present in Doc 2 but missing in Doc 1." },
            removedClauses: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Clauses present in Doc 1 but deleted/removed in Doc 2." },
            paymentDifferences: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of changes in costs, financial obligations, rates, fees, or penalties." },
            riskChanges: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Detailed summary list of how risks shifted between the draft versions." },
          },
        },
      },
    });

    const resultText = response.text;
    if (!resultText) {
      throw new Error("No comparison response returned from model.");
    }

    const parsedResult = JSON.parse(resultText.trim());
    res.json({
      ...parsedResult,
      comparedAt: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error("Comparison Error:", error);
    res.status(500).json({ error: error?.message || "An error occurred during comparison." });
  }
});

// 3. AI Chat Assistant & Q&A Endpoint
app.post("/api/chat", async (req, res) => {
  try {
    const { documentText, question, chatHistory, language } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Question is required." });
    }

    const ai = getGeminiClient();
    const systemInstruction = `You are LegalMind, an elite, friendly, and objective legal assistant designed to help average people understand complex legal documents.
You are given the full text of a legal document (listed below) and should use it strictly as context to answer the user's questions.

Guidelines:
1. Explain elements clearly using layperson phrasing.
2. Be objective, and do not provide binding formal representation. Include an educational reminder about obtaining a professional lawyer's opinion.
3. If they ask 'Should I sign this?' or 'Is this safe?', outline the yellow/red aspects clearly to help them make their own reasoned choice, rather than telling them a direct 'Yes' or 'No'.
4. Answer in simple, elegant language. If a translation language is active (${language || "English"}), write your final explanation response directly translated in that chosen language.`;

    // Reconstruct historic messages for the chat context
    // Limit to past 4 exchanges to keep token budget clean
    const recentHistory = (chatHistory || []).slice(-8);
    const contents: any[] = [];

    // Provide context as the first block
    contents.push({
      role: "user",
      parts: [{
        text: `Here is the legal document contract context:
---
${documentText || "No document loaded yet."}
---`
      }]
    });

    contents.push({
      role: "model",
      parts: [{ text: "Understood. I have read and indexed this legal document. What questions do you have about it? I will answer simply, clearly, and objectively." }]
    });

    // Populate historical conversational turns
    recentHistory.forEach((msg: any) => {
      contents.push({
        role: msg.sender === "user" ? "user" : "model",
        parts: [{ text: msg.text }],
      });
    });

    // Append the active question
    contents.push({
      role: "user",
      parts: [{ text: question }],
    });

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: contents,
      config: {
        systemInstruction: systemInstruction,
      },
    });

    const resultText = response.text;
    res.json({ message: resultText || "I failed to formulate an explanation. Could you please rephrase?" });
  } catch (error: any) {
    console.error("Chat Error:", error);
    res.status(500).json({ error: error?.message || "An error occurred during chat conversation." });
  }
});

// Vite Middleware & Static Asset Routing
async function setupServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in DEVELOPMENT mode with Vite Middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in PRODUCTION mode with static routing...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`LegalMind Backend running at http://0.0.0.0:${PORT}`);
  });
}

setupServer();

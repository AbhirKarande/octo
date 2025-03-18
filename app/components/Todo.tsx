export type Action = {
  thought: string;
  command: string;
  args: string[];
  uncertainties?: number[];
};

const handleSubmit = async (input: string) => {
  const action: Action = {
    thought: processedThought,
    command: processedCommand,
    args: processedArgs,
    uncertainties: modelOutput.confidenceScores
  };
};

{action.args.map((arg, index) => (
  <div key={index}>
    <span className="arg">{arg}</span>
    {action.uncertainties && (
      <span className="uncertainty">
        (Uncertainty: {Math.round(action.uncertainties[index] * 100)}%)
      </span>
    )}
  </div>
))} 